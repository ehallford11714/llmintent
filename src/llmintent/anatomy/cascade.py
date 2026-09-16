"""Three graph layers and nominated-unit cascade interventions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch

from llmintent.anatomy.component import (
    ablate_ffn_units,
    capture_ffn_input,
    mean_unit_activity,
    restore_ffn_units,
)
from llmintent.anatomy.evidence import UNIDENTIFIED
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
    a_changes_b: bool = False
    task_disrupted: bool = False
    task_restored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathway": list(self.pathway),
            "baseline": self.baseline,
            "intervene_a": self.intervene_a,
            "intervene_b": self.intervene_b,
            "rescue": self.rescue,
            "controls": self.controls,
            "verdict": self.verdict,
            "a_changes_b": self.a_changes_b,
            "task_disrupted": self.task_disrupted,
            "task_restored": self.task_restored,
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


def cascade_verdict(
    *,
    a_changes_b: bool,
    task_disrupted: bool,
    task_restored: bool,
    vs_random: bool,
) -> str:
    """Pathway claim requires A→B activity, task disruption, and restoration."""
    if a_changes_b and task_disrupted and task_restored and vs_random:
        return "intervention-supported pathway"
    if a_changes_b and (task_disrupted or vs_random):
        return "candidate cascade"
    return UNIDENTIFIED


def _action(bundle: Any, item: Any | None) -> float | None:
    if item is None:
        return None
    from llmintent.anatomy.tasks import forced_choice_logprobs

    return float(forced_choice_logprobs(bundle, item)["action_score"])


def _unidentified(*notes: str) -> CascadeResult:
    return CascadeResult(
        pathway=["unidentified"],
        baseline={},
        intervene_a={"source": UNIDENTIFIED},
        intervene_b={"source": UNIDENTIFIED},
        rescue=None,
        controls={},
        verdict=UNIDENTIFIED,
        notes=list(notes),
    )


def run_cascade(
    bundle: Any,
    cand_a: Any,
    cand_b: Any,
    prompt: str,
    *,
    task_item: Any | None = None,
    rng: np.random.Generator | None = None,
    activity_eps: float = 0.05,
) -> CascadeResult:
    """Intervene on nominated FFN units: A must change B, rescue must restore the task."""
    rng = rng or np.random.default_rng(0)
    if cand_a is None or cand_b is None:
        return _unidentified("Missing nominated candidates; random first-FFN modules were not substituted.")
    units_a = [int(u) for u in (getattr(cand_a, "unit_indices", None) or [])]
    units_b = [int(u) for u in (getattr(cand_b, "unit_indices", None) or [])]
    path_a = str(getattr(cand_a, "module_path", "") or "")
    path_b = str(getattr(cand_b, "module_path", "") or "")
    if not units_a or not units_b or not path_a or not path_b:
        return _unidentified("Nominated candidates lack FFN unit indices.")
    if path_a == path_b and set(units_a) & set(units_b):
        return _unidentified("A and B share units on the same module; cascade unidentified.")

    z_b_base = capture_ffn_input(bundle, prompt, path_b)
    b_base = mean_unit_activity(z_b_base, units_b)
    action_base = _action(bundle, task_item)

    with ablate_ffn_units(bundle, path_a, units_a, scale=0.0):
        z_b_under_a = capture_ffn_input(bundle, prompt, path_b)
        b_under_a = mean_unit_activity(z_b_under_a, units_b)
        action_a = _action(bundle, task_item)
        with no_op_hooks(bundle):
            pass

    restore_vals = {
        int(u): float(z_b_base[u])
        for u in units_b
        if 0 <= int(u) < z_b_base.size
    }
    with ablate_ffn_units(bundle, path_a, units_a, scale=0.0):
        with restore_ffn_units(bundle, path_b, restore_vals):
            z_b_rescue = capture_ffn_input(bundle, prompt, path_b)
            b_rescue = mean_unit_activity(z_b_rescue, units_b)
            action_rescue = _action(bundle, task_item)

    z_a = capture_ffn_input(bundle, prompt, path_a)
    pool = [i for i in range(int(z_a.size)) if i not in set(units_a)]
    take = min(len(units_a), len(pool))
    random_units = [int(i) for i in rng.choice(pool, size=take, replace=False)] if take else []
    with ablate_ffn_units(bundle, path_a, random_units, scale=0.0) if random_units else _noop():
        z_b_rand = capture_ffn_input(bundle, prompt, path_b)
        b_under_rand = mean_unit_activity(z_b_rand, units_b)
        action_rand = _action(bundle, task_item)

    a_delta = abs(b_under_a - b_base)
    a_changes_b = bool(a_delta > activity_eps * (abs(b_base) + 0.1))
    vs_random_activity = bool(a_delta > abs(b_under_rand - b_base) + 1e-9) if random_units else False

    task_disrupted = False
    task_restored = False
    vs_random_task = False
    if action_base is not None and action_a is not None:
        task_disrupted = abs(action_a - action_base) > activity_eps
        if action_rescue is not None:
            task_restored = abs(action_rescue - action_base) + 1e-9 < abs(action_a - action_base)
        if action_rand is not None:
            vs_random_task = abs(action_a - action_base) > abs(action_rand - action_base) + 1e-9

    vs_random = bool(vs_random_activity or vs_random_task)
    verdict = cascade_verdict(
        a_changes_b=a_changes_b,
        task_disrupted=task_disrupted,
        task_restored=task_restored,
        vs_random=vs_random,
    )
    return CascadeResult(
        pathway=[
            getattr(cand_a, "region_id", path_a),
            getattr(cand_b, "region_id", path_b),
            "action_score",
        ],
        baseline={
            "b_activity": b_base,
            "action_score": action_base,
            "source": "measured",
        },
        intervene_a={
            "b_activity": b_under_a,
            "action_score": action_a,
            "units": units_a,
            "module_path": path_a,
            "source": "measured",
        },
        intervene_b={
            "note": "B is measured under A, not independently steered in residual space.",
            "units": units_b,
            "module_path": path_b,
            "source": "measured",
        },
        rescue={
            "b_activity": b_rescue,
            "action_score": action_rescue,
            "source": "measured",
            "method": "clamp B units to baseline while A remains ablated",
        },
        controls={
            "random_units": random_units,
            "b_activity_random_a": b_under_rand,
            "action_score_random_a": action_rand,
            "vs_random": vs_random,
        },
        verdict=verdict,
        a_changes_b=a_changes_b,
        task_disrupted=task_disrupted,
        task_restored=task_restored,
        notes=[
            "Cascade intervenes on the nominated SVD units, not the first two FFN modules.",
            "intervention-supported pathway requires A→B activity change, task disruption, rescue, and a random-unit control.",
            "Rescue restores B's measured units; success is part of the verdict.",
            "Hooks removed after each condition.",
        ],
    )


@contextmanager
def _noop() -> Iterator[None]:
    yield
