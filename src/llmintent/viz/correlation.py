"""Correlation matrix visualizations for concepts and reasoning traces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from llmintent.trajectory import TrajectoryMapping
from llmintent.viz.backend import new_figure, save_figure


def build_concept_correlation_matrix(mapping: TrajectoryMapping) -> pd.DataFrame:
    """Pearson correlation between concept activation traces across layers."""
    cols = [c for c in mapping.layers.columns if c.endswith("_activation") or c.endswith("_similarity")]
    if len(cols) < 2:
        cols = [c for c in ["reasoning", "meta_reasoning", "ideation", "identity"] if c in mapping.layers.columns]
    if len(cols) < 2:
        cols = [
            c
            for c in [
                "entropy",
                "kl_divergence",
                "intensity",
                "occupancy",
                "motor_alignment",
                "top_intent_prob",
            ]
            if c in mapping.layers.columns
        ]
    if len(cols) < 2:
        raise ValueError("Need at least two concept or trace columns for correlation")
    return mapping.layers[cols].astype(float).corr(method="pearson")


def build_reasoning_trace_correlation(mapping: TrajectoryMapping) -> pd.DataFrame:
    """Correlation matrix of reasoning-related trajectory signals per layer."""
    trace_cols = [
        c
        for c in [
            "entropy",
            "entropy_drop",
            "kl_divergence",
            "kl_weight",
            "intensity",
            "occupancy",
            "motor_alignment",
            "reasoning",
            "meta_reasoning",
            "ideation",
            "barlow_invariance",
            "barlow_redundancy",
        ]
        if c in mapping.layers.columns
    ]
    if len(trace_cols) < 2:
        raise ValueError("Insufficient reasoning trace columns")
    return mapping.layers[trace_cols].astype(float).corr(method="pearson")


def build_cross_concept_layer_matrix(mapping: TrajectoryMapping) -> pd.DataFrame:
    """
    Layer × concept matrix (not correlation) for heatmap display.
    Rows = layers, cols = concept activation columns.
    """
    cols = [c for c in mapping.layers.columns if "concept_" in c and c.endswith("_activation")]
    if not cols:
        cols = [c for c in ["reasoning", "meta_reasoning", "ideation"] if c in mapping.layers.columns]
    return mapping.layers.set_index("layer")[cols].astype(float)


def plot_correlation_matrix(
    corr: pd.DataFrame,
    *,
    title: str = "Correlation Matrix",
    annot: bool = True,
    ax=None,
):
    """Seaborn-style correlation heatmap."""
    import seaborn as sns

    if ax is None:
        _, ax = new_figure(figsize=(max(8, len(corr) * 0.8), max(6, len(corr) * 0.7)))

    mask = np.eye(len(corr), dtype=bool) if len(corr) > 1 else None
    sns.heatmap(
        corr,
        ax=ax,
        annot=annot,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        mask=mask,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title(title)
    return ax


def plot_concept_correlation(
    mapping: TrajectoryMapping,
    *,
    title: str = "Concept Activation Correlation",
    ax=None,
):
    corr = build_concept_correlation_matrix(mapping)
    return plot_correlation_matrix(corr, title=title, ax=ax)


def plot_reasoning_trace_correlation(
    mapping: TrajectoryMapping,
    *,
    title: str = "Reasoning Trace Correlation",
    ax=None,
):
    corr = build_reasoning_trace_correlation(mapping)
    return plot_correlation_matrix(corr, title=title, ax=ax)


def save_concept_correlation(mapping: TrajectoryMapping, path: str, **kwargs) -> str:
    fig, ax = new_figure()
    plot_concept_correlation(mapping, ax=ax, **kwargs)
    return save_figure(fig, path)


def save_reasoning_trace_correlation(mapping: TrajectoryMapping, path: str, **kwargs) -> str:
    fig, ax = new_figure()
    plot_reasoning_trace_correlation(mapping, ax=ax, **kwargs)
    return save_figure(fig, path)
