"""Animations for trajectory maturation and reasoning subspace evolution."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import numpy as np
from sklearn.decomposition import PCA

from llmintent.forward import forward_hidden_states, normalize_hidden
from llmintent.jspace.trace import IntentTrace
from llmintent.models import ModelBundle
from llmintent.trajectory import TrajectoryMapping
from llmintent.viz.backend import new_figure


def animate_trajectory_maturation(
    mapping: TrajectoryMapping,
    *,
    metrics: list[str] | None = None,
    interval_ms: int = 400,
    save_path: str | None = None,
) -> animation.FuncAnimation:
    """
    Animate layer-by-layer maturation of trajectory metrics (line chart build-up).
    """
    import matplotlib.pyplot as plt

    metrics = metrics or [c for c in ["entropy", "kl_divergence", "intensity"] if c in mapping.layers.columns]
    if not metrics:
        raise ValueError("No metrics available for animation")

    df = mapping.layers
    layers = df["layer"].values
    fig, ax = new_figure()
    lines = {}
    for i, metric in enumerate(metrics):
        (line,) = ax.plot([], [], marker="o", label=metric, linewidth=2)
        lines[metric] = line

    ax.set_xlim(layers.min() - 0.5, layers.max() + 0.5)
    y_vals = df[metrics].astype(float).values.flatten()
    ax.set_ylim(y_vals.min() * 0.9, y_vals.max() * 1.1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.set_title(f"Trajectory Maturation — {mapping.model_name}")
    ax.legend(loc="upper right")

    pivot_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9, verticalalignment="top")

    def init():
        for line in lines.values():
            line.set_data([], [])
        pivot_text.set_text("")
        return list(lines.values()) + [pivot_text]

    def update(frame):
        idx = frame + 1
        for metric in metrics:
            lines[metric].set_data(layers[:idx], df[metric].values[:idx])
        active_pivots = [k for k, v in mapping.pivots.items() if v < idx]
        intent = df.iloc[min(frame, len(df) - 1)].get("top_intent", "")
        pivot_text.set_text(f"Layer {frame}\nTop intent: {intent!r}\nPivots seen: {', '.join(active_pivots)}")
        return list(lines.values()) + [pivot_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(df),
        init_func=init,
        blit=True,
        interval=interval_ms,
    )

    if save_path:
        _save_animation(anim, save_path)
        plt.close(fig)
    return anim


def animate_reasoning_subspace(
    bundle: ModelBundle,
    prompt: str,
    trace: IntentTrace | None = None,
    *,
    interval_ms: int = 500,
    save_path: str | None = None,
) -> animation.FuncAnimation:
    """Animate a point moving through the 2D PCA reasoning subspace layer by layer."""
    import matplotlib.pyplot as plt

    _, states = forward_hidden_states(bundle, prompt)
    vectors = np.stack(
        [normalize_hidden(bundle, s[0, -1, :].float().cpu()).numpy() for s in states]
    )
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    fig, ax = new_figure()
    ax.set_xlim(coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5)
    ax.set_ylim(coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Reasoning Subspace Animation")

    trail, = ax.plot([], [], color="gray", alpha=0.4, linewidth=1)
    point, = ax.plot([], [], "o", color="#C44E52", markersize=12)
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9, verticalalignment="top")

    def init():
        trail.set_data([], [])
        point.set_data([], [])
        label.set_text("")
        return trail, point, label

    def update(frame):
        trail.set_data(coords[: frame + 1, 0], coords[: frame + 1, 1])
        point.set_data([coords[frame, 0]], [coords[frame, 1]])
        thought = trace.top_thought_at(frame) if trace else ""
        regime = ""
        if trace and frame < len(trace.layer_stats):
            regime = str(trace.layer_stats.iloc[frame].get("regime", ""))
        label.set_text(f"Layer {frame}\nRegime: {regime}\nIntent: {thought!r}")
        return trail, point, label

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(coords),
        init_func=init,
        blit=True,
        interval=interval_ms,
    )

    if save_path:
        _save_animation(anim, save_path)
        plt.close(fig)
    return anim


def animate_intent_grid(
    trace: IntentTrace,
    *,
    position: int = -1,
    interval_ms: int = 600,
    save_path: str | None = None,
) -> animation.FuncAnimation:
    """Animate top intent at each layer for a fixed token position (layer thoughts filmstrip)."""
    import matplotlib.pyplot as plt

    pos = position if position >= 0 else trace.seq_len + position
    fig, ax = new_figure(figsize=(10, 4))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    title = ax.text(0, 0.3, "", ha="center", va="center", fontsize=14, fontweight="bold")
    subtitle = ax.text(0, -0.1, "", ha="center", va="center", fontsize=11)

    def init():
        title.set_text("")
        subtitle.set_text("")
        return title, subtitle

    def update(frame):
        intent = trace.top1_grid[frame][pos] if frame < len(trace.top1_grid) else ""
        regime = trace.layer_stats.iloc[frame].get("regime", "") if frame < len(trace.layer_stats) else ""
        title.set_text(f"Layer {frame}: {intent!r}")
        subtitle.set_text(f"Regime: {regime}  |  Position: {pos}")
        return title, subtitle

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=trace.num_layers,
        init_func=init,
        blit=True,
        interval=interval_ms,
    )

    if save_path:
        _save_animation(anim, save_path)
        plt.close(fig)
    return anim


def _save_animation(anim: animation.FuncAnimation, path: str) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".gif"):
        writer = animation.PillowWriter(fps=3)
        anim.save(path, writer=writer)
    elif path.endswith(".mp4"):
        try:
            anim.save(path, writer="ffmpeg", fps=3)
        except Exception:
            gif_path = path.replace(".mp4", ".gif")
            writer = animation.PillowWriter(fps=3)
            anim.save(gif_path, writer=writer)
    else:
        gif_path = path if path.endswith(".gif") else f"{path}.gif"
        writer = animation.PillowWriter(fps=3)
        anim.save(gif_path, writer=writer)
