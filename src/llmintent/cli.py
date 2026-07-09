"""Command-line interface for LLMIntent."""

from __future__ import annotations

import argparse
import json
import sys


def _add_model_args(p: argparse.ArgumentParser, *, model_required: bool = False) -> None:
    """Shared --model / --family / --size / --device flags."""
    p.add_argument(
        "--model",
        default=None,
        required=model_required,
        help="HF model id or suite key (family:size)",
    )
    p.add_argument(
        "--family",
        default=None,
        choices=["qwen", "mistral", "minimax", "glm", "legacy"],
        help="Model suite family (see: llmintent models list)",
    )
    p.add_argument(
        "--size",
        default=None,
        choices=["tiny", "small", "medium", "large", "xl"],
        help="Suite size tier (default: medium when --family is set)",
    )
    p.add_argument("--device", default=None, help="Torch device (or LLMINTENT_DEVICE)")


def _resolve_cli_model(args: argparse.Namespace, *, default: str = "gpt2") -> str:
    from llmintent.suite import resolve_model_id

    if getattr(args, "family", None):
        return resolve_model_id(
            model=getattr(args, "model", None),
            family=args.family,
            size=getattr(args, "size", None) or "medium",
        )
    if getattr(args, "model", None):
        return resolve_model_id(model=args.model)
    return resolve_model_id(default=default)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic extraction and intent analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    from llmintent.suite_cli import add_suite_parsers, handle_suite_command, maybe_patch_trajectory_parser

    add_suite_parsers(sub)

    analyze = sub.add_parser("analyze", help="Analyze a single prompt")
    _add_model_args(analyze)
    analyze.add_argument("--prompt", required=True)
    analyze.add_argument("--cot", default=None, help="Optional chain-of-thought prompt")
    analyze.add_argument("--compaction", action="store_true")
    analyze.add_argument("--blocks", action="store_true")

    compare = sub.add_parser("compare-cot", help="Compare direct vs CoT intensity")
    _add_model_args(compare)
    compare.add_argument("--direct", required=True)
    compare.add_argument("--cot", required=True)

    trace = sub.add_parser("trace", help="J-space intent trace and activation layers")
    _add_model_args(trace)
    trace.add_argument("--prompt", required=True)
    trace.add_argument("--transport", action="store_true", help="Fit J-lens transport maps")
    trace.add_argument("--track", nargs="*", default=[], help="Tokens to track rank across layers")

    layers = sub.add_parser("layers", help="Layer correspondence map")
    _add_model_args(layers)
    layers.add_argument("--prompt", required=True)
    layers.add_argument("--transport", action="store_true")

    cognitive = sub.add_parser("cognitive", help="Identity/reasoning/meta/ideation kernels")
    _add_model_args(cognitive)
    cognitive.add_argument("--twin-a", required=True)
    cognitive.add_argument("--twin-b", required=True)

    query = sub.add_parser("query", help="Query semantic concept in activation trajectory")
    _add_model_args(query)
    query.add_argument("--concept", required=True, help="Semantic concept text to locate")
    query.add_argument("--prompt", required=True, help="Anchor prompt for trajectory")
    query.add_argument("--twin-b", default=None, help="Twin prompt for KL-Barlow (defaults to --prompt)")
    query.add_argument("--top-k", type=int, default=5)

    trajectory = sub.add_parser(
        "trajectory",
        help="Activation trajectory (--prompt) or isolates reasoning trajectory (--text)",
    )
    _add_model_args(trajectory)
    trajectory.add_argument(
        "--prompt",
        default=None,
        help="Prompt for activation trajectory mapping (model path)",
    )
    trajectory.add_argument("--twin-b", default=None)
    trajectory.add_argument("--concepts", nargs="*", default=[], help="Concepts to annotate on trajectory")
    maybe_patch_trajectory_parser(trajectory)

    viz = sub.add_parser("viz", help="Visualization suite: maps, correlations, animations")
    _add_model_args(viz)
    viz.add_argument("--prompt", required=True)
    viz.add_argument("--twin-b", default=None)
    viz.add_argument("--concepts", nargs="*", default=[])
    viz.add_argument("--output-dir", default="llmintent_viz")
    viz.add_argument(
        "--type",
        choices=["full", "trajectory-map", "morpheme-map", "subspace", "concept-corr", "reasoning-corr", "trajectory-anim", "subspace-anim", "intent-anim"],
        default="full",
        help="Visualization type (default: full report)",
    )
    viz.add_argument("--blocks", action="store_true", help="Include morpheme map (requires GloVe)")
    viz.add_argument("--transport", action="store_true")

    heighten = sub.add_parser("heighten", help="Heightened reasoning via forced retrace")
    _add_model_args(heighten)
    heighten.add_argument("--prompt", required=True)
    heighten.add_argument("--anchor", default=None, help="CoT or anchor prompt (defaults to --prompt)")
    heighten.add_argument("--concepts", nargs="*", default=[])
    heighten.add_argument(
        "--mode",
        choices=["explicit_retrace", "concept_anchor", "pivot_replay", "correction", "focused_cot"],
        default="explicit_retrace",
    )
    heighten.add_argument("--steer", action="store_true", help="Apply activation focus steering")
    heighten.add_argument("--diagnose-only", action="store_true", help="Only report focus metrics")
    heighten.add_argument("--transport", action="store_true")

    benchmark = sub.add_parser("benchmark", help="HellaSwag SLM benchmark with retrace ablations")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_cmd", required=True)

    hs = benchmark_sub.add_parser("hellaswag", help="Run HellaSwag with retrace ablations")
    hs.add_argument("--models", nargs="+", default=["gpt2", "distilgpt2"])
    hs.add_argument("--limit", type=int, default=20)
    hs.add_argument("--full", action="store_true", help="Run full validation split (10,042 examples)")
    hs.add_argument("--conditions", default="fast", help="fast, default, or comma-separated")
    hs.add_argument("--store", default="llmintent_retraces/hellaswag.jsonl")
    hs.add_argument("--fallback", action="store_true", help="Use built-in fixture subset")
    hs.add_argument("--no-focus", action="store_true", help="Skip focus metric measurement")

    cmp = benchmark_sub.add_parser("compare", help="Summarize accuracy from retrace store")
    cmp.add_argument("--store", default="llmintent_retraces/hellaswag.jsonl")
    cmp.add_argument("--export-csv", default=None)

    slms = benchmark_sub.add_parser("slms", help="List prepared SLM targets")
    slms.set_defaults(benchmark_cmd="slms")

    retracement = sub.add_parser("retracement", help="Retracement Transformer perplexity & ablation")
    rt_sub = retracement.add_subparsers(dest="retracement_cmd", required=True)

    rt_ppl = rt_sub.add_parser("perplexity", help="Perplexity for one retracement mode")
    rt_ppl.add_argument("--model", default="gpt2")
    rt_ppl.add_argument(
        "--mode",
        choices=[
            "baseline",
            "focus_gate",
            "retrace_steer",
            "dual_pass",
            "workspace_loop",
            "extreme",
        ],
        default="focus_gate",
    )
    rt_ppl.add_argument("--limit", type=int, default=24)

    rt_ab = rt_sub.add_parser("ablation", help="Perplexity ablation across modes")
    rt_ab.add_argument("--models", nargs="+", default=["gpt2", "distilgpt2"])
    rt_ab.add_argument("--limit", type=int, default=24)
    rt_ab.add_argument("--full", action="store_true", help="All six modes (slower)")

    live = sub.add_parser("live", help="Real-time Live suite (Phi-3, Qwen, SLMs)")
    live_sub = live.add_subparsers(dest="live_cmd", required=True)

    live_models = live_sub.add_parser("models", help="List registered live models")

    live_serve = live_sub.add_parser("serve", help="Start FastAPI server")
    live_serve.add_argument("--model", default="qwen-0.5b")
    live_serve.add_argument("--host", default="127.0.0.1")
    live_serve.add_argument("--port", type=int, default=8765)

    live_ui = live_sub.add_parser("ui", help="Launch Streamlit app")

    live_run = live_sub.add_parser("run", help="One-shot analyze / heighten / generate")
    live_run.add_argument("--model", default="qwen-0.5b")
    live_run.add_argument("--prompt", required=True)
    live_run.add_argument(
        "--action",
        choices=["analyze", "heighten", "generate", "probe"],
        default="analyze",
    )
    live_run.add_argument("--retracement-mode", default="focus_gate")
    live_run.add_argument("--steer", action="store_true")
    live_run.add_argument("--max-tokens", type=int, default=64)

    models_cmd = sub.add_parser("models", help="List / inspect curated model suite")
    models_sub = models_cmd.add_subparsers(dest="models_cmd", required=True)
    models_list = models_sub.add_parser("list", help="List suite models")
    models_list.add_argument("--family", default=None, choices=["qwen", "mistral", "minimax", "glm", "legacy"])
    models_list.add_argument("--no-legacy", action="store_true")
    models_info = models_sub.add_parser("info", help="Show one suite entry")
    models_info.add_argument("family", choices=["qwen", "mistral", "minimax", "glm", "legacy"])
    models_info.add_argument("size", nargs="?", default="medium", choices=["tiny", "small", "medium", "large", "xl"])
    models_env = models_sub.add_parser("env", help="Show LLMINTENT_* env resolution")

    run_cmd = sub.add_parser("run", help="Quick text generation with a suite model")
    _add_model_args(run_cmd)
    run_cmd.add_argument("--text", "--prompt", dest="text", required=True, help="Prompt text")
    run_cmd.add_argument("--max-new-tokens", type=int, default=32)

    args = parser.parse_args(argv)

    suite_rc = handle_suite_command(args)
    if suite_rc is not None:
        return suite_rc

    if args.command == "models":
        from llmintent.suite import get_model_spec, list_models, resolve_from_env

        if args.models_cmd == "list":
            rows = list_models(family=args.family, include_legacy=not args.no_legacy)
            print(json.dumps(rows, indent=2))
            return 0
        if args.models_cmd == "info":
            print(json.dumps(get_model_spec(args.family, args.size).to_dict(), indent=2))
            return 0
        if args.models_cmd == "env":
            print(json.dumps(resolve_from_env(), indent=2))
            return 0

    if args.command == "run":
        from llmintent.suite import load_suite_model

        model_id = _resolve_cli_model(args, default="distilgpt2")
        bundle = load_suite_model(model=model_id, device=args.device)
        try:
            import torch

            tok = bundle.tokenizer
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            enc = tok(args.text, return_tensors="pt")
            enc = {k: v.to(bundle.device) for k, v in enc.items()}
            with torch.no_grad():
                out = bundle.model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            text = tok.decode(out[0], skip_special_tokens=True)
            print(
                json.dumps(
                    {"model": bundle.name, "prompt": args.text, "output": text},
                    indent=2,
                )
            )
        finally:
            del bundle.model
        return 0

    if args.command == "analyze":
        from llmintent import LLMIntentAnalyzer

        model_id = _resolve_cli_model(args)
        analyzer = LLMIntentAnalyzer(model_id, device=args.device)
        try:
            report = analyzer.analyze_prompt(
                args.prompt,
                cot_prompt=args.cot,
                include_compaction=args.compaction,
                include_block_semantics=args.blocks,
            )
            payload = {
                "model": report.model_name,
                "prompt": report.prompt,
                "intensity_peak_layer": int(report.intensity_sweep["intensity"].idxmax())
                if not report.intensity_sweep.empty
                else None,
                "intensity_peak": float(report.intensity_sweep["intensity"].max())
                if not report.intensity_sweep.empty
                else None,
                "entropy_trajectory": report.entropy_trajectory.to_dict(orient="records"),
                "cot_comparison": report.cot_comparison,
                "pivot_entropy": report.pivot_entropy,
                "inference_pivot": report.inference_pivot,
                "activation_layers": report.activation_layers,
            }
            print(json.dumps(payload, indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "compare-cot":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(_resolve_cli_model(args), device=args.device)
        try:
            sweep = analyzer.compare_prompts({"Direct": args.direct, "CoT": args.cot})
            cot = compare_cot_from_sweep(sweep)
            print(json.dumps(cot, indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "trace":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(
            _resolve_cli_model(args),
            device=args.device,
            load_glove=False,
            fit_jspace_transport=args.transport,
        )
        try:
            trace = analyzer.intent_trace(args.prompt, track_tokens=args.track or None)
            payload = {
                "prompt": trace.prompt,
                "activation_layers": trace.activation_layers,
                "regime_bands": trace.regime_bands,
                "entropy": trace.entropy,
                "occupancy": trace.occupancy,
                "rank_curves": trace.rank_curves,
                "layer_stats": trace.layer_stats.to_dict(orient="records"),
            }
            print(json.dumps(payload, indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "layers":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(
            _resolve_cli_model(args),
            device=args.device,
            load_glove=False,
            fit_jspace_transport=args.transport,
        )
        try:
            layer_map = analyzer.layer_correspondence(args.prompt)
            print(json.dumps(layer_map.to_dict(orient="records"), indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "cognitive":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(_resolve_cli_model(args), device=args.device, load_glove=False)
        try:
            profile = analyzer.cognitive_modules(args.twin_a, args.twin_b)
            print(json.dumps(profile.to_dict(), indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "query":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(_resolve_cli_model(args), device=args.device, load_glove=False)
        try:
            result = analyzer.query_concept(
                args.concept,
                args.prompt,
                twin_b=args.twin_b,
                top_k_layers=args.top_k,
            )
            print(json.dumps(result.to_dict(), indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "trajectory":
        if not args.prompt:
            print(
                "trajectory requires --prompt (activation map) or --text (isolates reasoning).",
                file=sys.stderr,
            )
            return 2
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(_resolve_cli_model(args), device=args.device, load_glove=False)
        try:
            mapping = analyzer.trajectory_map(
                args.prompt,
                twin_b=args.twin_b,
                concepts=args.concepts or None,
            )
            print(json.dumps(mapping.to_dict(), indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "viz":
        from llmintent import LLMIntentAnalyzer

        load_glove = args.blocks or args.type == "morpheme-map"
        analyzer = LLMIntentAnalyzer(
            _resolve_cli_model(args),
            device=args.device,
            load_glove=load_glove,
            fit_jspace_transport=args.transport,
        )
        try:
            suite = analyzer.visualizer(output_dir=args.output_dir)
            mapping = suite.trajectory_mapping(
                args.prompt,
                twin_b=args.twin_b,
                concepts=args.concepts or None,
            )
            trace = suite.intent_trace(args.prompt)
            paths: dict[str, str] = {}

            if args.type in ("full", "trajectory-map"):
                paths["trajectory_map"] = suite.save_trajectory_map(mapping)
            if args.type in ("full", "subspace"):
                paths["reasoning_subspace"] = suite.save_reasoning_subspace(
                    args.prompt, mapping=mapping
                )
            if args.type in ("full", "concept-corr"):
                paths["concept_correlation"] = suite.save_concept_correlation(mapping)
            if args.type in ("full", "reasoning-corr"):
                paths["reasoning_correlation"] = suite.save_reasoning_correlation(mapping)
            if args.type in ("full", "trajectory-anim"):
                paths["trajectory_animation"] = suite.save_trajectory_animation(mapping)
            if args.type in ("full", "subspace-anim"):
                paths["subspace_animation"] = suite.save_subspace_animation(
                    args.prompt, trace=trace
                )
            if args.type in ("full", "intent-anim"):
                paths["intent_animation"] = suite.save_intent_animation(trace)
            if args.type in ("full", "morpheme-map") or args.blocks:
                semantics = analyzer.extract_block_semantics()
                paths["morpheme_map"] = suite.save_morpheme_map(semantics)

            print(json.dumps({"output_dir": args.output_dir, "files": paths}, indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "heighten":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(
            _resolve_cli_model(args),
            device=args.device,
            load_glove=False,
            fit_jspace_transport=args.transport,
        )
        try:
            if args.diagnose_only:
                focus, mapping = analyzer.diagnose_focus(
                    args.prompt,
                    anchor_prompt=args.anchor,
                    concepts=args.concepts or None,
                )
                print(
                    json.dumps(
                        {
                            "focus": focus.to_dict(),
                            "pivots": mapping.pivots,
                        },
                        indent=2,
                    )
                )
            else:
                result = analyzer.heighten_reasoning(
                    args.prompt,
                    anchor_prompt=args.anchor,
                    concepts=args.concepts or None,
                    mode=args.mode,
                    apply_steering=args.steer,
                )
                print(json.dumps(result.to_dict(), indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "benchmark":
        if args.benchmark_cmd == "slms":
            from llmintent.benchmark import list_slms

            print(json.dumps(list_slms(), indent=2))
            return 0

        if args.benchmark_cmd == "compare":
            from llmintent.benchmark import RetraceStore

            store = RetraceStore(args.store)
            summary = store.summarize_accuracy()
            payload = {"summary": summary.to_dict(orient="records") if not summary.empty else []}
            if args.export_csv:
                path = store.export_csv(args.export_csv)
                payload["csv"] = path
            print(json.dumps(payload, indent=2))
            return 0

        if args.benchmark_cmd == "hellaswag":
            from llmintent.benchmark import BenchmarkRunConfig, HellaSwagBenchmarkRunner, parse_conditions

            config = BenchmarkRunConfig(
                models=args.models,
                conditions=parse_conditions(args.conditions),
                limit=10042 if args.full else args.limit,
                store_path=args.store,
                measure_focus=not args.no_focus,
                use_fallback=args.fallback,
            )
            runner = HellaSwagBenchmarkRunner(config)
            results = runner.run_all()
            compare = runner.compare_from_store()
            print(
                json.dumps(
                    {
                        "run_results": results.to_dict(orient="records"),
                        "store_summary": compare.to_dict(orient="records") if not compare.empty else [],
                        "store_path": args.store,
                    },
                    indent=2,
                )
            )
            return 0

    if args.command == "retracement":
        if args.retracement_cmd == "perplexity":
            from llmintent.retracement import RetracementConfig, RetracementMode, evaluate_perplexity, load_eval_texts

            cfg = RetracementConfig(mode=RetracementMode(args.mode))
            texts = load_eval_texts(limit=args.limit)
            result = evaluate_perplexity(args.model, cfg, texts)
            print(json.dumps(result.to_dict(), indent=2))
            return 0

        if args.retracement_cmd == "ablation":
            from llmintent.retracement import run_retracement_ablation

            df = run_retracement_ablation(
                models=args.models,
                fast=not args.full,
                text_limit=args.limit,
            )
            print(json.dumps(df.to_dict(orient="records"), indent=2))
            return 0

    if args.command == "live":
        if args.live_cmd == "models":
            from llmintent.live import list_live_models

            print(json.dumps(list_live_models(), indent=2))
            return 0

        if args.live_cmd == "serve":
            from llmintent.live.api import serve

            serve(host=args.host, port=args.port, model=args.model)
            return 0

        if args.live_cmd == "ui":
            import subprocess
            import sys
            from pathlib import Path

            import llmintent.live.ui as ui_mod

            script = Path(ui_mod.__file__).resolve()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(script),
                    "--server.port",
                    "8501",
                ],
                check=False,
            )
            return 0

        if args.live_cmd == "run":
            from llmintent.live import LiveIntentPipeline, LiveSessionConfig

            pipe = LiveIntentPipeline(
                LiveSessionConfig(
                    model_key=args.model,
                    retracement_mode=args.retracement_mode,
                )
            )
            try:
                pipe.load()
                if args.action == "analyze":
                    out = pipe.analyze(args.prompt)
                elif args.action == "heighten":
                    out = pipe.heighten(args.prompt, steer=args.steer)
                elif args.action == "generate":
                    out = pipe.generate(
                        args.prompt,
                        max_new_tokens=args.max_tokens,
                        retracement_mode=args.retracement_mode,
                        steer=args.steer,
                    )
                else:
                    tokens = pipe.probe_next_tokens(args.prompt, retracement_mode=args.retracement_mode)
                    out = {"tokens": [{"token": t, "prob": p} for t, p in tokens]}
                print(json.dumps(out.to_dict() if hasattr(out, "to_dict") else out, indent=2))
            finally:
                pipe.unload()
            return 0

    return 1


def compare_cot_from_sweep(sweep) -> dict:
    direct_total = float(sweep[sweep["prompt_type"] == "Direct"]["intensity"].sum())
    cot_total = float(sweep[sweep["prompt_type"] == "CoT"]["intensity"].sum())
    return {
        "direct_cumulative": direct_total,
        "cot_cumulative": cot_total,
        "work_ratio": cot_total / direct_total if direct_total else None,
        "layers": sweep.to_dict(orient="records"),
    }


if __name__ == "__main__":
    sys.exit(main())
