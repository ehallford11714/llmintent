"""Region A vs region B ablation.

The claim to test: driving the subspace assigned to region A, against the
subspace assigned to region B, changes the output. Synthetic path uses a
linear readout on planted axes (offline). Model path injects the SVD axis
at the region's depth band via residual-stream hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llmintent.anatomy.atlas import REGION_BY_ID, layers_for_region
from llmintent.anatomy.svd_map import SVDAnatomy, _unit, region_axis_from_anatomy


@dataclass
class AblationResult:
    region_a: str
    region_b: str
    handles_a: str
    handles_b: str
    top_a: list[str]
    top_b: list[str]
    kl_ab: float
    changed: bool
    gain: float
    method: str
    layers_a: list[int] = field(default_factory=list)
    layers_b: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "region_a": self.region_a,
            "region_b": self.region_b,
            "handles_a": self.handles_a,
            "handles_b": self.handles_b,
            "top_a": list(self.top_a),
            "top_b": list(self.top_b),
            "kl_ab": round(self.kl_ab, 6),
            "changed": self.changed,
            "gain": self.gain,
            "method": self.method,
            "layers_a": list(self.layers_a),
            "layers_b": list(self.layers_b),
            "notes": list(self.notes),
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def ablate_linear(
    hidden: np.ndarray,
    readout: np.ndarray,
    axes: dict[str, np.ndarray],
    region_a: str,
    region_b: str,
    *,
    gain: float = 3.0,
    token_names: list[str] | None = None,
) -> AblationResult:
    """
    h_A = h + gain * v_A − gain * v_B
    h_B = h + gain * v_B − gain * v_A
    logits = readout @ h  (readout is [vocab, hidden])
    """
    h = np.asarray(hidden, dtype=np.float64).reshape(-1)
    W = np.asarray(readout, dtype=np.float64)
    v_a = _unit(axes[region_a])
    v_b = _unit(axes[region_b])
    h_a = h + gain * v_a - gain * v_b
    h_b = h + gain * v_b - gain * v_a
    logits_a = W @ h_a
    logits_b = W @ h_b
    names = token_names or [str(i) for i in range(W.shape[0])]
    top_a = names[int(np.argmax(logits_a))]
    top_b = names[int(np.argmax(logits_b))]
    p_a, p_b = _softmax(logits_a), _softmax(logits_b)
    kl = _kl(p_a, p_b)
    k_a = min(3, len(names))
    return AblationResult(
        region_a=region_a,
        region_b=region_b,
        handles_a=REGION_BY_ID[region_a].handles,
        handles_b=REGION_BY_ID[region_b].handles,
        top_a=[names[i] for i in np.argsort(-logits_a)[:k_a]],
        top_b=[names[i] for i in np.argsort(-logits_b)[:k_a]],
        kl_ab=kl,
        changed=(top_a != top_b) or kl > 0.15,
        gain=gain,
        method="linear_readout",
        notes=[
            f"Top token A={top_a} vs B={top_b}.",
            "Planted or SVD axes; not a fly recording.",
        ],
    )


def plant_and_ablate(
    region_a: str = "vision",
    region_b: str = "auditory",
    *,
    dim: int = 32,
    gain: float = 4.0,
    seed: int = 0,
) -> tuple[AblationResult, dict[str, np.ndarray]]:
    """Construct orthogonal axes and a readout that prefers each region's token."""
    rng = np.random.default_rng(seed)
    names = [region_a, region_b, "other_0", "other_1"]
    axes = {
        region_a: _unit(np.eye(dim)[0]),
        region_b: _unit(np.eye(dim)[1]),
    }
    W = np.zeros((4, dim), dtype=np.float64)
    W[0, 0] = 1.0
    W[1, 1] = 1.0
    W[2, 2] = 0.4
    W[3, 3] = 0.4
    h = rng.normal(scale=0.05, size=dim)
    result = ablate_linear(
        h, W, axes, region_a, region_b, gain=gain, token_names=names
    )
    return result, axes


def ablate_model(
    bundle: Any,
    prompt: str,
    anatomy: SVDAnatomy,
    region_a: str,
    region_b: str,
    *,
    gain: float = 1.5,
    top_k: int = 5,
) -> AblationResult:
    """Steer residual stream along region A vs B SVD axes and compare next tokens."""
    import torch
    import torch.nn.functional as F

    from llmintent.forward import forward_hidden_states, get_lm_head, normalize_hidden
    from llmintent.heighten.intervention import steering_hooks
    from llmintent.models import get_unembedding_matrix

    n = int(bundle.num_layers)
    layers_a = layers_for_region(region_a, n)
    layers_b = layers_for_region(region_b, n)
    dim = int(bundle.hidden_size)
    v_a, src_a = region_axis_from_anatomy(anatomy, region_a, dim)
    v_b, src_b = region_axis_from_anatomy(anatomy, region_b, dim)
    if v_a is None or v_b is None:
        return AblationResult(
            region_a=region_a,
            region_b=region_b,
            handles_a=REGION_BY_ID[region_a].handles,
            handles_b=REGION_BY_ID[region_b].handles,
            top_a=[],
            top_b=[],
            kl_ab=0.0,
            changed=False,
            gain=gain,
            method="unidentified",
            layers_a=layers_a,
            layers_b=layers_b,
            notes=[
                f"Axis source A={src_a} B={src_b}. Missing axes are unidentified, not random fills.",
            ],
        )
    t_a = torch.from_numpy(v_a.astype(np.float32))
    t_b = torch.from_numpy(v_b.astype(np.float32))

    def _logits_with(vec: torch.Tensor, layers: list[int], coef: float) -> torch.Tensor:
        with steering_hooks(bundle, layers, vec, coef):
            _, states = forward_hidden_states(bundle, prompt)
        head = get_lm_head(bundle)
        last = normalize_hidden(bundle, states[-1][0, -1, :].float())
        try:
            return head(last.to(bundle.device)).detach().float().cpu()
        except Exception:
            unembed = get_unembedding_matrix(bundle.model).float()
            return F.linear(last.cpu(), unembed.cpu()).detach().float()

    logits_a = _logits_with(t_a, layers_a, gain) - _logits_with(t_b, layers_b, gain * 0.5)
    # Second pass for B (clean: only B). Re-run independently.
    logits_b_only = _logits_with(t_b, layers_b, gain)

    # Recompute A-only independently so comparison is A-drive vs B-drive.
    logits_a_only = _logits_with(t_a, layers_a, gain)
    _ = logits_a  # combined contrast kept for debugging hooks

    def _top(logits: torch.Tensor) -> list[str]:
        ids = torch.topk(logits, k=min(top_k, logits.numel())).indices.tolist()
        out = []
        for i in ids:
            piece = bundle.tokenizer.decode([i], skip_special_tokens=True).strip()
            out.append(piece or f"id:{i}")
        return out

    top_a = _top(logits_a_only)
    top_b = _top(logits_b_only)
    p_a = _softmax(logits_a_only.numpy())
    p_b = _softmax(logits_b_only.numpy())
    kl = _kl(p_a, p_b)
    return AblationResult(
        region_a=region_a,
        region_b=region_b,
        handles_a=REGION_BY_ID[region_a].handles,
        handles_b=REGION_BY_ID[region_b].handles,
        top_a=top_a,
        top_b=top_b,
        kl_ab=kl,
        changed=(top_a[:1] != top_b[:1]) or kl > 0.05,
        gain=gain,
        method="residual_steer",
        layers_a=layers_a,
        layers_b=layers_b,
        notes=[
            "Steered last-token residual at each region's depth band.",
            "Output change is a next-token distribution shift, not a behaviour proof.",
        ],
    )
