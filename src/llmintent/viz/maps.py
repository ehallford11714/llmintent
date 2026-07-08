"""Map visualizations: morphemes, trajectories, reasoning subspaces."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from llmintent.forward import forward_hidden_states, normalize_hidden
from llmintent.models import ModelBundle
from llmintent.trajectory import TrajectoryMapping
from llmintent.viz.backend import MODULE_COLORS, REGIME_COLORS, new_figure, save_figure


def plot_morpheme_map(
    block_semantics: dict[int, dict[str, list[str]]],
    *,
    top_morphemes: int = 20,
    title: str = "Morpheme Map: Layer × Semantic Unit",
    ax=None,
):
    """
    Heatmap of morpheme presence across transformer layers and components.

    block_semantics: output of LLMIntentAnalyzer.extract_block_semantics()
    """
    morpheme_layers: Counter[str] = Counter()
    rows: list[dict] = []

    for layer_idx, components in block_semantics.items():
        for component, units in components.items():
            for unit in units[:10]:
                morpheme_layers[unit] += 1
                rows.append({"layer": layer_idx, "morpheme": unit, "component": component, "present": 1.0})

    if not rows:
        raise ValueError("No morpheme data to plot")

    top = [m for m, _ in morpheme_layers.most_common(top_morphemes)]
    df = pd.DataFrame(rows)
    pivot = (
        df[df["morpheme"].isin(top)]
        .groupby(["layer", "morpheme"])["present"]
        .max()
        .unstack(fill_value=0)
    )

    if ax is None:
        _, ax = new_figure(figsize=(max(10, len(top) * 0.45), max(6, len(pivot) * 0.35)))

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"L{i}" for i in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Semantic unit (morpheme/lemma)")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="Present")
    return ax


def plot_trajectory_map(
    mapping: TrajectoryMapping,
    *,
    metrics: list[str] | None = None,
    title: str | None = None,
    ax=None,
):
    """Heatmap of activation trajectory metrics across layers."""
    metrics = metrics or [
        c
        for c in [
            "entropy",
            "kl_divergence",
            "intensity",
            "occupancy",
            "motor_alignment",
            "reasoning",
            "meta_reasoning",
            "ideation",
        ]
        if c in mapping.layers.columns
    ]
    if not metrics:
        raise ValueError("No plottable metrics in trajectory mapping")

    data = mapping.layers.set_index("layer")[metrics].astype(float)
    normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)

    if ax is None:
        _, ax = new_figure(figsize=(max(8, len(metrics) * 1.2), max(6, len(data) * 0.3)))

    im = ax.imshow(normalized.T.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xticks(range(len(data.index)))
    ax.set_xticklabels([f"L{i}" for i in data.index], rotation=45, ha="right")
    ax.set_xlabel("Layer")
    ax.set_title(title or f"Activation Trajectory — {mapping.model_name}")

    # Mark pivots
    for pivot_name, layer in mapping.pivots.items():
        if layer in data.index:
            idx = list(data.index).index(layer)
            ax.axvline(idx, color="white", linestyle="--", alpha=0.6, linewidth=0.8)

    ax.figure.colorbar(im, ax=ax, label="Normalized value")
    return ax


def plot_reasoning_subspace(
    bundle: ModelBundle,
    prompt: str,
    *,
    layer_stats: pd.DataFrame | None = None,
    title: str = "Reasoning Subspace (PCA of layer hidden states)",
    ax=None,
):
    """
    2D PCA map of per-layer hidden states colored by regime or cognitive module.
    """
    _, states = forward_hidden_states(bundle, prompt)
    vectors = np.stack(
        [normalize_hidden(bundle, s[0, -1, :].float().cpu()).numpy() for s in states]
    )
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    if ax is None:
        _, ax = new_figure()

    colors = []
    labels = []
    for i in range(len(states)):
        if layer_stats is not None and i < len(layer_stats):
            if "dominant_module" in layer_stats.columns:
                mod = str(layer_stats.iloc[i].get("dominant_module", "unknown"))
                colors.append(MODULE_COLORS.get(mod, "#888888"))
                labels.append(mod)
            elif "regime" in layer_stats.columns:
                reg = str(layer_stats.iloc[i].get("regime", "unknown"))
                colors.append(REGIME_COLORS.get(reg, "#888888"))
                labels.append(reg)
            else:
                colors.append("#888888")
                labels.append("")
        else:
            colors.append("#4C72B0")
            labels.append("")

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, edgecolors="black", linewidth=0.5)
    for i, (x, y) in enumerate(coords):
        ax.annotate(f"L{i}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)

    # Trajectory path
    ax.plot(coords[:, 0], coords[:, 1], color="gray", alpha=0.4, linewidth=1, zorder=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)
    return ax, coords


def save_morpheme_map(block_semantics, path: str, **kwargs) -> str:
    fig, ax = new_figure()
    plot_morpheme_map(block_semantics, ax=ax, **kwargs)
    return save_figure(fig, path)


def save_trajectory_map(mapping: TrajectoryMapping, path: str, **kwargs) -> str:
    fig, ax = new_figure()
    plot_trajectory_map(mapping, ax=ax, **kwargs)
    return save_figure(fig, path)


def save_reasoning_subspace(bundle, prompt, path: str, **kwargs) -> str:
    fig, ax = new_figure()
    plot_reasoning_subspace(bundle, prompt, ax=ax, **kwargs)
    return save_figure(fig, path)
