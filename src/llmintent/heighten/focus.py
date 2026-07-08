"""Focused reasoning metrics from trajectory and cognitive signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from llmintent.heighten.types import FocusMetrics
from llmintent.trajectory import TrajectoryMapping


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.sum() <= 0:
        return 1.0
    p = values / values.sum()
    p = p[p > 0]
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    max_ent = float(np.log(len(values) + 1e-12))
    return ent / max(max_ent, 1e-8)


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den <= 1e-8:
        return default
    return num / den


def compute_focus_metrics(
    mapping: TrajectoryMapping,
    *,
    focus_threshold: float = 0.45,
) -> FocusMetrics:
    """
    Score how focused reasoning is for a prompt trajectory.

    Unfocused patterns:
    - Reasoning signal spread across many layers (low concentration)
    - High ideation relative to reasoning
    - Elevated meta_reasoning (monitoring without commit)
    - Concept activations diffuse rather than peaked
    - Motor alignment rising before reasoning peak (premature commit)
    """
    df = mapping.layers.copy()
    n = len(df)

    reasoning = df["reasoning"].astype(float).values if "reasoning" in df else np.zeros(n)
    ideation = df["ideation"].astype(float).values if "ideation" in df else np.zeros(n)
    meta = df["meta_reasoning"].astype(float).values if "meta_reasoning" in df else np.zeros(n)
    motor = df["motor_alignment"].astype(float).values if "motor_alignment" in df else np.zeros(n)

    reasoning_concentration = 1.0 - _normalized_entropy(np.clip(reasoning, 0, None))

    concept_cols = [c for c in df.columns if c.endswith("_activation")]
    if concept_cols:
        concept_matrix = df[concept_cols].astype(float).values
        peakiness = []
        for row in concept_matrix:
            row = np.clip(row, 0, None)
            peakiness.append(_safe_ratio(float(row.max()), float(row.mean()), 0.0))
        concept_peakiness = float(np.mean(peakiness)) if peakiness else 0.5
    else:
        concept_peakiness = reasoning_concentration

    reasoning_sum = float(reasoning.sum()) + 1e-8
    ideation_sum = float(ideation.sum()) + 1e-8
    reasoning_ideation_ratio = _safe_ratio(reasoning_sum, ideation_sum, reasoning_sum)

    meta_load = float(meta.mean()) if meta.size else 0.0

    if "dominant_module" in df.columns:
        modules = df["dominant_module"].astype(str).tolist()
        counts = np.array([modules.count(m) for m in set(modules)], dtype=float)
        dispersion_index = _normalized_entropy(counts)
    else:
        dispersion_index = 0.5

    reasoning_peak = int(np.argmax(reasoning)) if reasoning.size else n // 2
    motor_early = float(motor[: max(1, reasoning_peak)].mean()) if motor.size else 0.0
    motor_late = float(motor[reasoning_peak:].mean()) if motor.size else 0.0
    motor_prematurity = max(0.0, motor_early - motor_late * 0.5)

    focus_score = float(
        0.30 * reasoning_concentration
        + 0.25 * min(concept_peakiness / 3.0, 1.0)
        + 0.20 * min(reasoning_ideation_ratio / 2.0, 1.0)
        + 0.15 * (1.0 - dispersion_index)
        + 0.10 * (1.0 - min(meta_load, 1.0))
        - 0.10 * min(motor_prematurity, 1.0)
    )
    focus_score = float(np.clip(focus_score, 0.0, 1.0))

    unfocused_layers: list[int] = []
    for i, row in df.iterrows():
        layer = int(row.get("layer", i))
        if "dominant_module" in row and row["dominant_module"] in ("ideation", "meta_reasoning"):
            unfocused_layers.append(layer)
        elif reasoning.size and reasoning[layer] < reasoning.max() * 0.25 and ideation[layer] > ideation.mean():
            unfocused_layers.append(layer)

    recommended = list(mapping.pivots.values())
    if reasoning.size:
        top_reason = np.argsort(reasoning)[-3:][::-1].tolist()
        recommended = sorted(set(recommended + top_reason))

    return FocusMetrics(
        reasoning_concentration=reasoning_concentration,
        concept_peakiness=concept_peakiness,
        reasoning_ideation_ratio=reasoning_ideation_ratio,
        meta_load=meta_load,
        dispersion_index=dispersion_index,
        motor_prematurity=motor_prematurity,
        focus_score=focus_score,
        needs_retrace=focus_score < focus_threshold,
        dominant_unfocused_layers=unfocused_layers[:8],
        recommended_focus_layers=recommended[:6],
    )


def compare_focus(before: FocusMetrics, after: FocusMetrics) -> dict[str, float]:
    """Delta metrics after retrace / heightening."""
    return {
        "focus_score_delta": after.focus_score - before.focus_score,
        "reasoning_concentration_delta": after.reasoning_concentration - before.reasoning_concentration,
        "concept_peakiness_delta": after.concept_peakiness - before.concept_peakiness,
        "meta_load_delta": before.meta_load - after.meta_load,
        "dispersion_delta": before.dispersion_index - after.dispersion_index,
        "retrace_recommended_resolved": float(not after.needs_retrace and before.needs_retrace),
    }
