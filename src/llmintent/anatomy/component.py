"""Nominated FFN-unit capture and intervention.

SVD names units. A functional claim requires the actual last-token
coefficient ``a_k = s_k v_k · z`` and an intervention on those units,
not a residual-stream substitute at a different module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from llmintent.anatomy.weights import get_module


@dataclass
class ComponentTrace:
    module_path: str
    layer: int | None
    singular_value: float
    coefficient: float
    units: list[int]
    unit_activations: list[float]
    prompt: str
    z_dim: int
    source: str = "measured"

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "layer": self.layer,
            "singular_value": self.singular_value,
            "coefficient": self.coefficient,
            "units": list(self.units),
            "unit_activations": list(self.unit_activations),
            "prompt": self.prompt,
            "z_dim": self.z_dim,
            "source": self.source,
        }


def capture_ffn_input(bundle: Any, prompt: str, module_path: str) -> np.ndarray:
    """Last-token pre-activation ``z`` at an FFN down-projection."""
    from llmintent.forward import forward_hidden_states

    captured: list[np.ndarray] = []
    module = get_module(bundle.model, module_path)

    def _pre(_mod, inputs):
        hidden = inputs[0]
        captured.append(hidden[0, -1].detach().float().cpu().numpy().reshape(-1))

    handle = module.register_forward_pre_hook(_pre)
    try:
        forward_hidden_states(bundle, prompt)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"FFN hook missed {module_path}")
    return np.asarray(captured[-1], dtype=np.float64)


def capture_ffn_coefficient(
    bundle: Any,
    prompt: str,
    candidate: Any,
) -> ComponentTrace:
    """``a_k = s_k v_k · z`` on the nominated component."""
    path = str(candidate.module_path)
    z = capture_ffn_input(bundle, prompt, path)
    vector = np.asarray(getattr(candidate, "v", None), dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("candidate has no SVD vector v")
    if vector.size != z.size:
        raise ValueError(f"v dim {vector.size} != FFN input {z.size}")
    scale = float(getattr(candidate, "singular_value", 0.0) or 0.0)
    units = [int(u) for u in (getattr(candidate, "unit_indices", None) or [])]
    return ComponentTrace(
        module_path=path,
        layer=getattr(candidate, "layer", None),
        singular_value=scale,
        coefficient=float(scale * np.dot(vector, z)),
        units=units,
        unit_activations=[float(z[u]) for u in units if 0 <= u < z.size],
        prompt=prompt,
        z_dim=int(z.size),
    )


@contextmanager
def ablate_ffn_units(
    bundle: Any,
    module_path: str,
    units: Sequence[int],
    *,
    scale: float = 0.0,
) -> Iterator[None]:
    """Scale nominated FFN units at the down-projection input (inference only)."""
    module = get_module(bundle.model, module_path)
    idx = [int(u) for u in units]
    if not idx:
        yield
        return

    def _pre(_mod, inputs):
        hidden = inputs[0].clone()
        hidden[..., idx] = hidden[..., idx] * scale
        return (hidden,) + tuple(inputs[1:])

    handle = module.register_forward_pre_hook(_pre)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def restore_ffn_units(
    bundle: Any,
    module_path: str,
    values: dict[int, float],
) -> Iterator[None]:
    """Clamp nominated units to recorded baseline activations."""
    module = get_module(bundle.model, module_path)
    if not values:
        yield
        return

    def _pre(_mod, inputs):
        hidden = inputs[0].clone()
        for index, value in values.items():
            if 0 <= int(index) < hidden.shape[-1]:
                hidden[..., int(index)] = float(value)
        return (hidden,) + tuple(inputs[1:])

    handle = module.register_forward_pre_hook(_pre)
    try:
        yield
    finally:
        handle.remove()


def mean_unit_activity(z: np.ndarray, units: Sequence[int]) -> float:
    idx = [int(u) for u in units if 0 <= int(u) < z.size]
    if not idx:
        return float("nan")
    return float(np.mean(z[idx]))


def nominated_intervention_test(
    bundle: Any,
    candidate: Any,
    items: Sequence[Any],
    control_items: Sequence[Any] | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Ablate nominated units vs matched-count random units on hold-out items."""
    from llmintent.anatomy.tasks import forced_choice_logprobs

    rng = rng or np.random.default_rng(0)
    units = [int(u) for u in (getattr(candidate, "unit_indices", None) or [])]
    path = str(candidate.module_path)
    control_items = list(control_items or [])

    def _scores(batch: Sequence[Any]) -> list[float]:
        return [float(forced_choice_logprobs(bundle, item)["action_score"]) for item in batch]

    baseline = _scores(items)
    control_base = _scores(control_items) if control_items else []
    with ablate_ffn_units(bundle, path, units, scale=0.0):
        ablated = _scores(items)
        control_abl = _scores(control_items) if control_items else []

    z_dim = 0
    try:
        z_dim = int(capture_ffn_input(bundle, items[0].prompt, path).size)
    except Exception:
        z_dim = max(units) + 1 if units else 0
    random_units = []
    if z_dim and units:
        pool = [i for i in range(z_dim) if i not in set(units)]
        take = min(len(units), len(pool))
        if take:
            random_units = [int(i) for i in rng.choice(pool, size=take, replace=False)]
    with ablate_ffn_units(bundle, path, random_units, scale=0.0) if random_units else _nullcontext():
        random_scores = _scores(items) if random_units else list(baseline)

    def _mean(vals: Iterable[float]) -> float:
        seq = list(vals)
        return float(np.mean(seq)) if seq else 0.0

    drop = _mean(baseline) - _mean(ablated)
    random_drop = _mean(baseline) - _mean(random_scores)
    control_drop = _mean(control_base) - _mean(control_abl) if control_items else 0.0
    selective = drop > control_drop and drop > random_drop
    return {
        "module_path": path,
        "layer": getattr(candidate, "layer", None),
        "units": units,
        "random_units": random_units,
        "baseline_action": _mean(baseline),
        "ablated_action": _mean(ablated),
        "random_ablated_action": _mean(random_scores),
        "control_baseline_action": _mean(control_base) if control_items else None,
        "control_ablated_action": _mean(control_abl) if control_items else None,
        "task_drop": drop,
        "control_drop": control_drop,
        "random_drop": random_drop,
        "selective": bool(selective),
        "source": "measured",
        "note": (
            "Nominated-unit intervention vs matched-count random units and "
            "control items. Not a biological validation."
        ),
    }


@contextmanager
def _nullcontext() -> Iterator[None]:
    yield
