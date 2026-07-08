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
