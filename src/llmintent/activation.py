"""Identify layers of peak activation and inference pivots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from llmintent.forward import forward_hidden_states
from llmintent.metrics import cosine_intensity
from llmintent.models import ModelBundle
from llmintent.poles import build_numerical_pole


def identify_activation_layers(
    bundle: ModelBundle,
    prompt: str,
    *,
    layer_stats: pd.DataFrame | None = None,
    entropy: list[float] | None = None,
    occupancy: list[float] | None = None,
    position: int = -1,
) -> dict[str, int]:
    """
    Detect key activation layers for a prompt.

    Returns layer indices for:
    - inference_pivot: largest entropy drop (maturation point)
    - workspace_peak: max J-space occupancy in workspace band
    - motor_onset: where decode aligns with final layer output
    - intensity_peak: max numerical-pole cosine similarity (notebook metric)
    """
    _, states = forward_hidden_states(bundle, prompt)
    num_layers = len(states)

    if entropy is None:
        from llmintent.steering import get_entropy_trajectory

        ent_df = get_entropy_trajectory(bundle, prompt)
        entropy = ent_df["entropy"].tolist()

    ent = np.array(entropy, dtype=float)
    ent_diff = np.diff(ent)
    inference_pivot = int(np.argmin(ent_diff) + 1) if len(ent_diff) else 0

    if occupancy is None:
        from llmintent.jspace.decompose import jspace_occupancy

        occupancy = [
            float(jspace_occupancy(bundle, states[i][0, position, :]))
            for i in range(num_layers)
        ]
    occ = np.array(occupancy, dtype=float)

    # Workspace band ~ middle third
    ws_start = num_layers // 3
    ws_end = (2 * num_layers) // 3
    ws_slice = occ[ws_start : ws_end + 1]
    workspace_peak = ws_start + int(np.argmax(ws_slice)) if len(ws_slice) else ws_start

    if layer_stats is not None and "motor_alignment" in layer_stats.columns:
        motor_onset = int(layer_stats["motor_alignment"].idxmax())
    else:
        motor_onset = max(0, num_layers - 2)

    pole = build_numerical_pole(bundle)
    intensities = [
        cosine_intensity(states[i][0, position, :], pole.to(bundle.device))
        for i in range(num_layers)
    ]
    intensity_peak = int(np.argmax(intensities))

    return {
        "inference_pivot": inference_pivot,
        "workspace_peak": workspace_peak,
        "motor_onset": motor_onset,
        "intensity_peak": intensity_peak,
    }


def activation_summary(
    bundle: ModelBundle,
    prompt: str,
    *,
    position: int = -1,
) -> pd.DataFrame:
    """Per-layer activation metrics for layer correspondence mapping."""
    from llmintent.jspace.decompose import jspace_occupancy
    from llmintent.steering import get_entropy_trajectory

    _, states = forward_hidden_states(bundle, prompt)
    ent_df = get_entropy_trajectory(bundle, prompt)
    pole = build_numerical_pole(bundle)

    rows: list[dict] = []
    for i, state in enumerate(states):
        hidden = state[0, position, :]
        rows.append(
            {
                "layer": i,
                "entropy": float(ent_df.loc[i, "entropy"]),
                "occupancy": float(jspace_occupancy(bundle, hidden)),
                "intensity": cosine_intensity(hidden, pole.to(bundle.device)),
            }
        )
    df = pd.DataFrame(rows)
    df["entropy_drop"] = -df["entropy"].diff().fillna(0)
    return df
