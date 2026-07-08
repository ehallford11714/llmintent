"""Classify transformer layers into sensory / workspace / motor regimes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class LayerRegime(str, Enum):
    SENSORY = "sensory"
    WORKSPACE = "workspace"
    MOTOR = "motor"


@dataclass(frozen=True)
class RegimeBands:
    sensory: tuple[int, int]
    workspace: tuple[int, int]
    motor: tuple[int, int]


def classify_layer_regimes(
    layer_stats: pd.DataFrame,
    *,
    num_layers: int,
) -> pd.DataFrame:
    """
    Assign Anthropic-style functional bands from per-layer decode statistics.

    Expected columns (any subset used if present):
    - layer, interpretability, top1_stability, entropy, occupancy, motor_alignment
    """
    work = layer_stats.copy()
    if "layer" not in work.columns:
        work["layer"] = range(len(work))

    # Build interpretability score from available signals
    score = np.zeros(len(work))
    weights = 0.0

    if "interpretability" in work.columns:
        score += work["interpretability"].to_numpy()
        weights += 1.0
    if "top1_stability" in work.columns:
        score += work["top1_stability"].to_numpy()
        weights += 1.0
    if "entropy" in work.columns:
        ent = work["entropy"].to_numpy()
        score += (ent.max() - ent) / (ent.max() - ent.min() + 1e-8)
        weights += 1.0
    if "motor_alignment" in work.columns:
        score += work["motor_alignment"].to_numpy()
        weights += 1.0

    if weights > 0:
        score /= weights
    else:
        score = np.linspace(0, 1, len(work))

    work["regime_score"] = score
    n = len(work)
    sensory_end = max(1, n // 3)
    motor_start = max(sensory_end + 1, (2 * n) // 3)

    regimes: list[str] = []
    for i in range(n):
        if i < sensory_end:
            regimes.append(LayerRegime.SENSORY.value)
        elif i >= motor_start:
            regimes.append(LayerRegime.MOTOR.value)
        else:
            regimes.append(LayerRegime.WORKSPACE.value)
    work["regime"] = regimes
    work["normalized_depth"] = work["layer"] / max(num_layers, 1)
    return work


def regime_bands(classified: pd.DataFrame) -> RegimeBands:
    """Return inclusive layer index ranges for each regime."""
    bands: dict[str, list[int]] = {r.value: [] for r in LayerRegime}
    for _, row in classified.iterrows():
        bands[row["regime"]].append(int(row["layer"]))

    def _span(values: list[int]) -> tuple[int, int]:
        if not values:
            return (0, 0)
        return (min(values), max(values))

    return RegimeBands(
        sensory=_span(bands[LayerRegime.SENSORY.value]),
        workspace=_span(bands[LayerRegime.WORKSPACE.value]),
        motor=_span(bands[LayerRegime.MOTOR.value]),
    )
