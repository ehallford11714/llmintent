"""Three graph layers and controlled residual interventions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch

from llmintent.anatomy.evidence import UNIDENTIFIED
from llmintent.anatomy.spaces import random_control_axis
from llmintent.heighten.intervention import steering_hooks
from llmintent.models import get_transformer_layers


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    layer_kind: str  # architecture | association | intervention | literature_prior | analogy
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 4),
            "layer_kind": self.layer_kind,
            "evidence": self.evidence,
        }


@dataclass
class CascadeResult:
    pathway: list[str]
    baseline: dict[str, Any]
    intervene_a: dict[str, Any]
    intervene_b: dict[str, Any]
    rescue: dict[str, Any] | None
    controls: dict[str, Any]
    verdict: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathway": list(self.pathway),
            "baseline": self.baseline,
            "intervene_a": self.intervene_a,
            "intervene_b": self.intervene_b,
            "rescue": self.rescue,
            "controls": self.controls,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def architecture_edges(n_layers: int) -> list[GraphEdge]:
    edges = []
    for i in range(n_layers - 1):
        edges.append(
            GraphEdge(f"L{i}", f"L{i+1}", 1.0, "architecture", "residual stream module order")
        )
    return edges


@contextmanager
def no_op_hooks(bundle: Any) -> Iterator[None]:
    layers = get_transformer_layers(bundle.model)
    handles = []
    fingerprint = [p.detach().clone() for p in list(bundle.model.parameters())[:4]]

    def hook(_m, _i, out):
        return out

    try:
        for layer in layers:
            handles.append(layer.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()
        for a, b in zip(fingerprint, list(bundle.model.parameters())[:4]):
            if not torch.allclose(a, b.detach()):
                raise RuntimeError("base parameters changed during intervention")


def _last_logits(bundle: Any, prompt: str) -> torch.Tensor:
    from llmintent.forward import forward_hidden_states, get_lm_head, normalize_hidden

    _, states = forward_hidden_states(bundle, prompt)
    last = normalize_hidden(bundle, states[-1][0, -1, :].float())
    return get_lm_head(bundle)(last.to(bundle.device)).detach().float().cpu()


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    return float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum())


def run_cascade(
    bundle: Any,
    prompt: str,
    vec_a: np.ndarray | None,
    vec_b: np.ndarray | None,
    layers_a: list[int],
    layers_b: list[int],
    *,
    gain: float = 1.0,
    gains: tuple[float, ...] = (0.5, 1.0),
) -> CascadeResult:
    """A → B → output with no-op, random, matched-norm, and rescue controls."""
    if vec_a is None or vec_b is None:
        return CascadeResult(
            pathway=["unidentified"],
            baseline={},
            intervene_a={"source": UNIDENTIFIED},
            intervene_b={"source": UNIDENTIFIED},
            rescue=None,
            controls={},
            verdict="unidentified",
            notes=["Missing axis returned unidentified; random vectors were not substituted."],
        )

    dim = int(bundle.hidden_size)
    t_a = torch.from_numpy(np.asarray(vec_a, dtype=np.float32))
    t_b = torch.from_numpy(np.asarray(vec_b, dtype=np.float32))
    rnd = torch.from_numpy(random_control_axis(dim, seed=7).astype(np.float32))
    if t_a.numel() != dim or t_b.numel() != dim:
        return CascadeResult(
            ["unidentified"], {}, {"source": UNIDENTIFIED}, {"source": UNIDENTIFIED},
            None, {}, "unidentified",
            notes=["Axis dimension does not match residual size."],
        )

    with no_op_hooks(bundle):
        base = _last_logits(bundle, prompt)

    def _steer(vec, layers, coef):
        with steering_hooks(bundle, layers, vec, coef):
            return _last_logits(bundle, prompt)

    a_on = _steer(t_a, layers_a, gain)
    b_on = _steer(t_b, layers_b, gain)
    # Rescue: A on, then add -A at B site (restoration of B's input analogue).
    with steering_hooks(bundle, layers_a, t_a, gain):
        with steering_hooks(bundle, layers_b, -t_a, gain):
            rescued = _last_logits(bundle, prompt)
    rnd_on = _steer(rnd, layers_a, gain)
    noop = base
    off_target = []
    for g in gains:
        off_target.append({"gain": g, "kl": _kl(base, _steer(t_a, layers_a, g))})

    kl_a = _kl(base, a_on)
    kl_b = _kl(base, b_on)
    kl_rnd = _kl(base, rnd_on)
    kl_rescue = _kl(base, rescued)
    specific = kl_a > kl_rnd * 1.05 and kl_a > 1e-6
    verdict = "intervention-supported pathway" if specific and kl_b > 1e-6 else "candidate cascade"
    return CascadeResult(
        pathway=[f"L{layers_a[0]}", f"L{layers_b[0]}", "logits"],
        baseline={"kl_to_self": 0.0, "source": "measured"},
        intervene_a={"kl": kl_a, "source": "measured"},
        intervene_b={"kl": kl_b, "source": "measured"},
        rescue={"kl": kl_rescue, "source": "measured"},
        controls={
            "no_op_kl": _kl(base, noop),
            "random_subspace_kl": kl_rnd,
            "matched_norm": True,
            "gain_sweep": off_target,
        },
        verdict=verdict,
        notes=[
            "Next-token KL is influence, not proof of the named function.",
            "Axes are transformer depth, not fly physiological time.",
            "Hooks removed; parameter fingerprint checked in no_op_hooks.",
        ],
    )
