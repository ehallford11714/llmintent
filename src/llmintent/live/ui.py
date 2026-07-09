"""Streamlit UI for LLMIntent Live — run with: llmintent live ui"""

from __future__ import annotations

from llmintent import __version__
from llmintent.live.pipeline import LiveIntentPipeline
from llmintent.live.registry import LIVE_MODELS, list_live_models
from llmintent.live.session import LiveSessionConfig
from llmintent.live.viz_panel import render_realtime_activation_stream, render_visualization_tab

# Bump when LiveIntentPipeline API changes so stale session instances are cleared.
_PIPELINE_API_VERSION = 2
def _parse_concepts(raw: str, prompt: str) -> list[str]:
    if raw.strip():
        parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
        return parts[:8]
    words = [w.strip(".,!?;:") for w in prompt.split() if len(w) > 3]
    return list(dict.fromkeys(words))[-5:] or ["reasoning", "answer"]


def run_streamlit() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Live UI requires streamlit. Install with: pip install llmintent[live]"
        ) from exc

    st.set_page_config(
        page_title="LLMIntent Live",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("LLMIntent Live")
    st.caption(f"Real-time focused reasoning · v{__version__}")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
        st.session_state.model_key = "gpt2"
        st.session_state.retrace_mode = "focus_gate"

    if st.session_state.get("pipeline_api_version") != _PIPELINE_API_VERSION:
        if st.session_state.pipeline is not None:
            try:
                st.session_state.pipeline.unload()
            except Exception:
                pass
        st.session_state.pipeline = None
        st.session_state.pipeline_api_version = _PIPELINE_API_VERSION

    with st.sidebar:
        st.header("Model")
        keys = [m.key for m in LIVE_MODELS]
        idx = keys.index(st.session_state.model_key) if st.session_state.model_key in keys else 0
        model_key = st.selectbox("Target model", keys, index=idx)
        retrace_mode = st.selectbox(
            "Retracement mode",
            ["baseline", "focus_gate", "dual_pass", "retrace_steer", "workspace_loop", "extreme"],
            index=["baseline", "focus_gate", "dual_pass", "retrace_steer", "workspace_loop", "extreme"].index(
                st.session_state.retrace_mode
            )
            if st.session_state.retrace_mode
            in ("baseline", "focus_gate", "dual_pass", "retrace_steer", "workspace_loop", "extreme")
            else 1,
        )
        st.session_state.retrace_mode = retrace_mode

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Load", type="primary", use_container_width=True):
                cfg = LiveSessionConfig(model_key=model_key, retracement_mode=retrace_mode)
                pipe = LiveIntentPipeline(cfg)
                with st.spinner(f"Loading {model_key}…"):
                    pipe.load()
                st.session_state.pipeline = pipe
                st.session_state.model_key = model_key
                st.success("Ready")
        with col_b:
            if st.session_state.pipeline and st.button("Unload", use_container_width=True):
                st.session_state.pipeline.unload()
                st.session_state.pipeline = None
                st.rerun()

        if st.session_state.pipeline:
            spec = st.session_state.pipeline.session.spec
            bundle = st.session_state.pipeline.session.bundle
            st.divider()
            st.markdown("**Loaded**")
            st.code(spec.hf_name, language=None)
            st.caption(f"Layers: {bundle.num_layers} · Device: {bundle.device}")

        st.divider()
        st.markdown("**Registry**")
        for m in list_live_models()[:4]:
            st.caption(f"`{m['key']}` — {m['params_m']:.0f}M")

    if st.session_state.pipeline is None:
        st.info("Select a model in the sidebar and click **Load** to start.")
        st.subheader("Available models")
        st.dataframe(
            [
                {
                    "key": m.key,
                    "hf_name": m.hf_name,
                    "params_m": m.params_m,
                    "chat": m.chat_template,
                    "default_mode": m.default_retracement_mode,
                }
                for m in LIVE_MODELS
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            """
            ### What you can do
            | Tab | Action |
            |-----|--------|
            | **Analyze** | Activation pivots + focus score |
            | **Heighten** | Forced retrace + focus gain |
            | **Generate** | Completion with Retracement Transformer |
            | **Compare** | Baseline vs retracement next-token probe |
            | **Probe** | Top-k next tokens |
            | **Visualize** | Real-time layer activation stream + maps, correlations, animations |
            """
        )
        return

    pipe: LiveIntentPipeline = st.session_state.pipeline
    if pipe.session.config.retracement_mode != retrace_mode:
        pipe.session.set_retracement_mode(retrace_mode)

    prompt = st.text_area(
        "Prompt",
        height=100,
        placeholder="Ask a question, paste HellaSwag context, or enter a reasoning task…",
    )
    concepts_raw = st.text_input(
        "Concepts (optional, space-separated)",
        placeholder="Leave empty to auto-infer from prompt",
    )
    concepts = _parse_concepts(concepts_raw, prompt) if prompt else []

    if concepts:
        st.caption(f"Concepts: {', '.join(concepts)}")

    tab_analyze, tab_heighten, tab_gen, tab_compare, tab_probe, tab_viz = st.tabs(
        ["Analyze", "Heighten", "Generate", "Compare", "Probe", "Visualize"]
    )

    with tab_analyze:
        c1, c2 = st.columns([1, 3])
        with c1:
            run_a = st.button("Run analyze", key="btn_analyze", use_container_width=True)
        if run_a and prompt:
            with st.spinner("Analyzing…"):
                result = pipe.analyze(prompt, concepts=concepts or None)
            m1, m2, m3 = st.columns(3)
            m1.metric("Focus", f"{result.focus_score:.3f}" if result.focus_score is not None else "—")
            m2.metric("Needs retrace", "Yes" if result.needs_retrace else "No" if result.needs_retrace is not None else "—")
            m3.metric("Latency", f"{result.latency_ms:.0f} ms")
            st.subheader("Activation pivots")
            st.json(result.activation_layers)
            if result.recommended_focus_layers:
                st.caption(f"Recommended steer layers: {result.recommended_focus_layers}")

    with tab_heighten:
        mode = st.selectbox(
            "Retrace mode",
            ["explicit_retrace", "concept_anchor", "pivot_replay", "correction", "focused_cot"],
            index=0,
        )
        steer = st.checkbox("Steering probe", value=False)
        if st.button("Run heighten", key="btn_heighten") and prompt:
            with st.spinner("Heightening…"):
                result = pipe.heighten(
                    prompt,
                    concepts=concepts or None,
                    mode=mode,
                    steer=steer,
                )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Before", f"{result.focus_before:.3f}" if result.focus_before is not None else "—")
            c2.metric("After", f"{result.focus_after:.3f}" if result.focus_after is not None else "—")
            c3.metric("Gain", f"{result.focus_gain:+.3f}" if result.focus_gain is not None else "—")
            c4.metric("ms", f"{result.latency_ms:.0f}")
            st.subheader("Retrace scaffold")
            st.code(result.retrace_prompt)
            focused = pipe.focused_prefix(prompt, concepts=concepts or None)
            with st.expander("Focused scaffold"):
                st.code(focused)
            if result.top_logits_shift:
                st.subheader("Logit shifts (steer)")
                st.bar_chart(result.top_logits_shift)

    with tab_gen:
        g1, g2, g3 = st.columns(3)
        max_tok = g1.slider("Max tokens", 4, 64, 16)
        temp = g2.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        gen_steer = g3.checkbox("Steer", value=False)
        live_layers = st.checkbox("Live layer activation stream", value=False)
        stream_tokens = st.slider("Stream max tokens", 4, 64, 16) if live_layers else 16
        if st.button("Generate", key="btn_gen", type="primary") and prompt:
            if live_layers:
                st.subheader("Live layer activation")
                snapshots = render_realtime_activation_stream(
                    st,
                    pipe,
                    prompt,
                    max_new_tokens=stream_tokens,
                    temperature=temp,
                    retracement_mode=retrace_mode,
                )
                if snapshots:
                    st.subheader("Completion")
                    st.write(snapshots[-1].cumulative_text)
                    st.caption(f"{retrace_mode} · {len(snapshots) - 1} tokens streamed")
            else:
                with st.spinner("Generating…"):
                    result = pipe.generate(
                        prompt,
                        max_new_tokens=max_tok,
                        temperature=temp,
                        retracement_mode=retrace_mode,
                        steer=gen_steer,
                    )
                st.subheader("Completion")
                st.write(result.completion)
                st.caption(
                    f"{result.latency_ms:.0f} ms · {result.retracement_mode} · steered={result.steered}"
                )

    with tab_compare:
        st.caption("Compare next-token distribution: baseline vs retracement mode")
        if st.button("Compare modes", key="btn_compare") and prompt:
            with st.spinner("Probing…"):
                base = pipe.probe_next_tokens(prompt, k=8, retracement_mode="baseline")
                retr = pipe.probe_next_tokens(prompt, k=8, retracement_mode=retrace_mode)
            left, right = st.columns(2)
            with left:
                st.markdown("**Baseline**")
                for tok, prob in base:
                    st.progress(min(prob * 4, 1.0), text=f"{tok!r} ({prob:.4f})")
            with right:
                st.markdown(f"**{retrace_mode}**")
                for tok, prob in retr:
                    st.progress(min(prob * 4, 1.0), text=f"{tok!r} ({prob:.4f})")

    with tab_probe:
        k = st.slider("Top-k", 3, 20, 8)
        if st.button("Probe", key="btn_probe") and prompt:
            tokens = pipe.probe_next_tokens(prompt, k=k, retracement_mode=retrace_mode)
            st.bar_chart({t: p for t, p in tokens})

    with tab_viz:
        st.caption("Real-time per-layer activation and full visualization suite.")

        st.subheader("Real-time layer activation")
        rt1, rt2, rt3 = st.columns(3)
        rt_max = rt1.slider("Stream tokens", 4, 64, 16, key="rt_max")
        rt_temp = rt2.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key="rt_temp")
        rt_mode = rt3.selectbox(
            "Mode",
            ["snapshot", "stream"],
            index=0,
            help="Snapshot = one forward pass on prompt. Stream = update each generated token.",
        )
        if st.button("Run live activation", key="btn_rt_viz", type="primary") and prompt:
            render_realtime_activation_stream(
                st,
                pipe,
                prompt,
                max_new_tokens=rt_max,
                temperature=rt_temp,
                retracement_mode=retrace_mode,
                snapshot_only=(rt_mode == "snapshot"),
            )

        st.divider()
        st.subheader("Full visualization report")
        twin_prompt = st.text_input(
            "Twin / CoT prompt (optional — enables KL divergence & cognitive traces)",
            placeholder="Let's think step by step. The capital of France is",
            help="Must differ from the main prompt. Example: prepend CoT scaffolding to the same question.",
        )
        v1, v2, v3 = st.columns(3)
        show_maps = v1.checkbox("Maps", value=True)
        show_corr = v2.checkbox("Correlations", value=True)
        show_anim = v3.checkbox("Animations (slow)", value=False)
        if st.button("Generate visualizations", key="btn_viz", type="primary") and prompt:
            render_visualization_tab(
                st,
                pipe,
                prompt,
                twin_b=twin_prompt.strip() or None,
                concepts=concepts if concepts_raw.strip() else None,
                include_concepts=bool(concepts_raw.strip()),
                include_maps=show_maps,
                include_correlations=show_corr,
                include_animations=show_anim,
            )


if __name__ == "__main__":
    run_streamlit()
