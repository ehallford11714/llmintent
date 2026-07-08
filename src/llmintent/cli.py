"""Command-line interface for LLMIntent."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic extraction and intent analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a single prompt")
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--prompt", required=True)
    analyze.add_argument("--cot", default=None, help="Optional chain-of-thought prompt")
    analyze.add_argument("--compaction", action="store_true")
    analyze.add_argument("--blocks", action="store_true")

    compare = sub.add_parser("compare-cot", help="Compare direct vs CoT intensity")
    compare.add_argument("--model", required=True)
    compare.add_argument("--direct", required=True)
    compare.add_argument("--cot", required=True)

    trace = sub.add_parser("trace", help="J-space intent trace and activation layers")
    trace.add_argument("--model", required=True)
    trace.add_argument("--prompt", required=True)
    trace.add_argument("--transport", action="store_true", help="Fit J-lens transport maps")
    trace.add_argument("--track", nargs="*", default=[], help="Tokens to track rank across layers")

    layers = sub.add_parser("layers", help="Layer correspondence map")
    layers.add_argument("--model", required=True)
    layers.add_argument("--prompt", required=True)
    layers.add_argument("--transport", action="store_true")

    cognitive = sub.add_parser("cognitive", help="Identity/reasoning/meta/ideation kernels")
    cognitive.add_argument("--model", required=True)
    cognitive.add_argument("--twin-a", required=True)
    cognitive.add_argument("--twin-b", required=True)

    query = sub.add_parser("query", help="Query semantic concept in activation trajectory")
    query.add_argument("--model", required=True)
    query.add_argument("--concept", required=True, help="Semantic concept text to locate")
    query.add_argument("--prompt", required=True, help="Anchor prompt for trajectory")
    query.add_argument("--twin-b", default=None, help="Twin prompt for KL-Barlow (defaults to --prompt)")
    query.add_argument("--top-k", type=int, default=5)

    trajectory = sub.add_parser("trajectory", help="Unified activation trajectory mapping")
    trajectory.add_argument("--model", required=True)
    trajectory.add_argument("--prompt", required=True)
    trajectory.add_argument("--twin-b", default=None)
    trajectory.add_argument("--concepts", nargs="*", default=[], help="Concepts to annotate on trajectory")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(args.model)
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

        analyzer = LLMIntentAnalyzer(args.model)
        try:
            sweep = analyzer.compare_prompts({"Direct": args.direct, "CoT": args.cot})
            cot = compare_cot_from_sweep(sweep)
            print(json.dumps(cot, indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "trace":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(args.model, load_glove=False, fit_jspace_transport=args.transport)
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

        analyzer = LLMIntentAnalyzer(args.model, load_glove=False, fit_jspace_transport=args.transport)
        try:
            layer_map = analyzer.layer_correspondence(args.prompt)
            print(json.dumps(layer_map.to_dict(orient="records"), indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "cognitive":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(args.model, load_glove=False)
        try:
            profile = analyzer.cognitive_modules(args.twin_a, args.twin_b)
            print(json.dumps(profile.to_dict(), indent=2))
        finally:
            analyzer.cleanup()
        return 0

    if args.command == "query":
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(args.model, load_glove=False)
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
        from llmintent import LLMIntentAnalyzer

        analyzer = LLMIntentAnalyzer(args.model, load_glove=False)
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
