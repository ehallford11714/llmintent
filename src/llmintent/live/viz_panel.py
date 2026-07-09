"""Streamlit visualization panel — embeds LLMIntent VisualizationSuite in Live UI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from llmintent.live.activation_stream import (
    LayerActivationSnapshot,
    iter_generate_with_layer_activations,
    snapshot_layer_activations,
)
from llmintent.live.generate import format_prompt
from llmintent.trajectory import TrajectoryMapping, build_trajectory_mapping
from llmintent.viz import (
    VisualizationSuite,
    plot_concept_correlation,
    plot_reasoning_subspace,
    plot_reasoning_trace_correlation,
    plot_trajectory_map,
)
from llmintent.viz.backend import new_figure

if TYPE_CHECKING:
    from llmintent.live.pipeline import LiveIntentPipeline

import pandas as pd
import numpy as np


def _snapshot_prompt_layers(pipe: LiveIntentPipeline, prompt: str) -> pd.DataFrame:
    """Snapshot layers — works even if session holds a pre-upgrade pipeline instance."""
    if hasattr(pipe, "snapshot_prompt_layers"):
        return pipe.snapshot_prompt_layers(prompt)
    bundle = pipe.session.bundle
    text = format_prompt(bundle, prompt, use_chat=pipe.session.spec.chat_template)
    enc = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
    return snapshot_layer_activations(pipe.session, enc.input_ids)


def _stream_layer_activations(pipe: LiveIntentPipeline, prompt: str, **kwargs):
    if hasattr(pipe, "stream_layer_activations"):
        yield from pipe.stream_layer_activations(prompt, **kwargs)
        return
    yield from iter_generate_with_layer_activations(pipe.session, prompt, **kwargs)


def _show_figure(st: Any, ax) -> None:
    fig = ax.figure if hasattr(ax, "figure") else ax
    st.pyplot(fig)
    plt.close(fig)


def _show_gif(st: Any, path: str) -> None:
    st.image(path, caption=Path(path).name)


def plot_live_activation_heatmap(
    metrics: pd.DataFrame,
    *,
    step: int = 0,
    token: str = "",
    ax=None,
):
    """Heatmap of per-layer entropy and intensity for one decoding step."""
    cols = [c for c in ["entropy", "intensity"] if c in metrics.columns]
    if not cols:
        raise ValueError("Need entropy and/or intensity columns")

    data = metrics.set_index("layer")[cols].astype(float)
    normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)

    if ax is None:
        _, ax = new_figure(figsize=(6, max(4, len(data) * 0.25)))

    im = ax.imshow(normalized.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f"L{i}" for i in data.index])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_ylabel("Layer")
    label = f" — token {token!r}" if token else " — prompt"
    ax.set_title(f"Live activation (step {step}){label}")
    ax.figure.colorbar(im, ax=ax, label="Normalized")
    return ax


def plot_live_entropy_profile(metrics: pd.DataFrame, *, step: int = 0, token: str = "", ax=None):
    """Line chart of entropy across layers for one step."""
    if ax is None:
        _, ax = new_figure(figsize=(8, 4))
    layers = metrics["layer"].values
    ax.plot(layers, metrics["entropy"].values, marker="o", linewidth=2, label="entropy")
    if "intensity" in metrics.columns:
        ax2 = ax.twinx()
        ax2.plot(
            layers,
            metrics["intensity"].values,
            marker="s",
            color="#55A868",
            linewidth=1.5,
            alpha=0.8,
            label="intensity",
        )
        ax2.set_ylabel("Intensity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy")
    label = f" — {token!r}" if token else " — prompt"
    ax.set_title(f"Layer profile (step {step}){label}")
    ax.grid(True, alpha=0.3)
    return ax


def plot_activation_timeline(snapshots: list[LayerActivationSnapshot], *, metric: str = "entropy", ax=None):
    """Layers × decoding steps heatmap for one metric across the stream."""
    if not snapshots:
        raise ValueError("No snapshots to plot")
    steps = [s.step for s in snapshots]
    layers = snapshots[0].layer_metrics["layer"].tolist()
    matrix = np.zeros((len(layers), len(snapshots)))
    for j, snap in enumerate(snapshots):
        col = snap.layer_metrics.set_index("layer")[metric]
        for i, layer in enumerate(layers):
            matrix[i, j] = float(col.get(layer, 0.0))
    normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)

    if ax is None:
        _, ax = new_figure(figsize=(max(8, len(snapshots) * 0.5), max(5, len(layers) * 0.3)))

    im = ax.imshow(normalized, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{i}" for i in layers])
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right")
    ax.set_xlabel("Decoding step")
    ax.set_ylabel("Layer")
    ax.set_title(f"Real-time {metric} maturation")
    ax.figure.colorbar(im, ax=ax, label="Normalized")
    return ax


def render_realtime_activation_stream(
    st: Any,
    pipe: LiveIntentPipeline,
    prompt: str,
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.7,
    retracement_mode: str | None = None,
    snapshot_only: bool = False,
) -> list[LayerActivationSnapshot]:
    """
    Stream per-token layer activations into Streamlit placeholders.

    When ``snapshot_only`` is True, renders a single prompt snapshot without generating.
    """
    if not prompt.strip():
        st.warning("Enter a prompt first.")
        return []

    status = st.empty()
    text_out = st.empty()
    chart_col, line_col = st.columns(2)
    heatmap_ph = chart_col.empty()
    profile_ph = line_col.empty()
    timeline_ph = st.empty()

    snapshots: list[LayerActivationSnapshot] = []

    if snapshot_only:
        with st.spinner("Capturing layer activations…"):
            metrics = _snapshot_prompt_layers(pipe, prompt)
        snap = LayerActivationSnapshot(
            step=0,
            token="",
            token_id=-1,
            layer_metrics=metrics,
            cumulative_text=prompt,
        )
        snapshots.append(snap)
        fig_h, ax_h = new_figure()
        plot_live_activation_heatmap(metrics, step=0, ax=ax_h)
        heatmap_ph.pyplot(fig_h)
        plt.close(fig_h)
        fig_l, ax_l = new_figure()
        plot_live_entropy_profile(metrics, step=0, ax=ax_l)
        profile_ph.pyplot(fig_l)
        plt.close(fig_l)
        status.caption(f"Snapshot · {len(metrics)} layers")
        return snapshots

    progress = st.progress(0.0, text="Starting activation stream…")
    for snap in _stream_layer_activations(
        pipe,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        retracement_mode=retracement_mode,
    ):
        snapshots.append(snap)
        step_label = "prompt" if snap.step == 0 else snap.token
        progress.progress(
            min(snap.step / max(max_new_tokens, 1), 1.0),
            text=f"Step {snap.step}/{max_new_tokens} · {step_label!r}",
        )
        fig_h, ax_h = new_figure()
        plot_live_activation_heatmap(
            snap.layer_metrics,
            step=snap.step,
            token=snap.token,
            ax=ax_h,
        )
        heatmap_ph.pyplot(fig_h)
        plt.close(fig_h)

        fig_l, ax_l = new_figure()
        plot_live_entropy_profile(
            snap.layer_metrics,
            step=snap.step,
            token=snap.token,
            ax=ax_l,
        )
        profile_ph.pyplot(fig_l)
        plt.close(fig_l)

        if snap.step == 0:
            text_out.caption(f"Prompt loaded · {len(snap.layer_metrics)} layers")
        else:
            text_out.write(snap.cumulative_text)

    progress.progress(1.0, text="Stream complete")
    status.success(f"Captured {len(snapshots)} activation snapshots across {max_new_tokens} max tokens")

    if len(snapshots) > 1:
        fig_t, ax_t = new_figure(figsize=(10, 5))
        plot_activation_timeline(snapshots, metric="entropy", ax=ax_t)
        timeline_ph.pyplot(fig_t)
        plt.close(fig_t)
        timeline_ph.caption("Entropy maturation across decoding steps")

    return snapshots


def build_live_mapping(
    pipe: LiveIntentPipeline,
    prompt: str,
    *,
    twin_b: str | None = None,
    concepts: list[str] | None = None,
    include_cognitive: bool = True,
    include_concepts: bool = True,
) -> TrajectoryMapping:
    """Build full trajectory mapping for visualization (may be slower than analyze path)."""
    bundle = pipe.session.bundle
    return build_trajectory_mapping(
        bundle,
        prompt,
        twin_b=twin_b,
        concepts=concepts if include_concepts else None,
        include_cognitive=include_cognitive,
        include_concepts=include_concepts,
    )


def render_visualization_tab(
    st: Any,
    pipe: LiveIntentPipeline,
    prompt: str,
    *,
    twin_b: str | None = None,
    concepts: list[str] | None = None,
    include_concepts: bool = False,
    include_maps: bool = True,
    include_correlations: bool = True,
    include_animations: bool = False,
) -> dict[str, str]:
    """
    Render visualization suite artifacts inline in Streamlit.

    Returns dict of artifact name → file path (for animations) or 'inline' for static plots.
    """
    if not prompt.strip():
        st.warning("Enter a prompt first.")
        return {}

    bundle = pipe.session.bundle
    out_dir = Path(tempfile.mkdtemp(prefix="llmintent_live_viz_"))
    suite = VisualizationSuite(bundle, output_dir=str(out_dir))

    twin = twin_b.strip() if twin_b and twin_b.strip() else None
    use_twin = bool(twin and twin != prompt.strip())
    if twin and not use_twin:
        st.info("Twin prompt matches main prompt — KL / cognitive traces need a different CoT twin.")

    mapping: TrajectoryMapping | None = None
    trace = None
    with st.spinner("Building trajectory mapping…"):
        try:
            mapping = build_live_mapping(
                pipe,
                prompt,
                twin_b=twin if use_twin else None,
                concepts=concepts if include_concepts else None,
                include_cognitive=use_twin,
                include_concepts=include_concepts and bool(concepts),
            )
            suite._mapping = mapping
            trace = suite.intent_trace(prompt)
            suite._trace = trace
        except Exception as exc:
            st.warning(f"Full mapping failed ({exc}). Retrying with lightweight path…")
            mapping = build_live_mapping(
                pipe,
                prompt,
                twin_b=None,
                concepts=None,
                include_cognitive=False,
                include_concepts=False,
            )
            suite._mapping = mapping
            trace = suite.intent_trace(prompt)
            suite._trace = trace

    if mapping is None:
        st.error("Could not build trajectory mapping.")
        return {}

    artifacts: dict[str, str] = {}

    if include_maps:
        st.subheader("Maps")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Trajectory heatmap**")
            fig, ax = new_figure()
            plot_trajectory_map(mapping, ax=ax)
            _show_figure(st, ax)
            artifacts["trajectory_map"] = "inline"
        with c2:
            st.markdown("**Reasoning subspace (PCA)**")
            fig, ax = new_figure()
            result = plot_reasoning_subspace(
                bundle,
                prompt,
                layer_stats=mapping.layers,
                ax=ax,
            )
            ax = result[0] if isinstance(result, tuple) else result
            _show_figure(st, ax)
            artifacts["reasoning_subspace"] = "inline"

    if include_correlations:
        st.subheader("Correlations")
        c1, c2 = st.columns(2)
        with c1:
            try:
                st.markdown("**Concept / trace correlation**")
                fig, ax = new_figure()
                plot_concept_correlation(mapping, ax=ax)
                _show_figure(st, ax)
                artifacts["concept_correlation"] = "inline"
            except ValueError as exc:
                st.caption(str(exc))
        with c2:
            try:
                st.markdown("**Reasoning signal correlation**")
                fig, ax = new_figure()
                plot_reasoning_trace_correlation(mapping, ax=ax)
                _show_figure(st, ax)
                artifacts["reasoning_correlation"] = "inline"
            except ValueError as exc:
                st.caption(str(exc))

    if include_animations:
        st.subheader("Animations")
        st.caption("GIF generation may take 30–60 seconds on SLMs.")
        a1, a2, a3 = st.columns(3)
        with a1:
            try:
                path = suite.save_trajectory_animation(mapping)
                _show_gif(st, path)
                artifacts["trajectory_animation"] = path
            except Exception as exc:
                st.caption(f"Trajectory anim: {exc}")
        with a2:
            try:
                path = suite.save_subspace_animation(prompt, trace=trace)
                _show_gif(st, path)
                artifacts["subspace_animation"] = path
            except Exception as exc:
                st.caption(f"Subspace anim: {exc}")
        with a3:
            try:
                path = suite.save_intent_animation(trace)
                _show_gif(st, path)
                artifacts["intent_animation"] = path
            except Exception as exc:
                st.caption(f"Intent anim: {exc}")

    with st.expander("Trajectory table (layers)"):
        st.dataframe(mapping.layers, use_container_width=True, hide_index=True)

    kl_max = float(mapping.layers["kl_divergence"].max()) if "kl_divergence" in mapping.layers else 0.0
    if use_twin:
        st.caption(f"Pivots: {mapping.pivots} · KL peak: {kl_max:.4f}")
    else:
        st.caption(
            f"Pivots: {mapping.pivots} · Add a **different** twin / CoT prompt above for KL & cognitive traces"
        )
    return artifacts
