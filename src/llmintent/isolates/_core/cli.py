"""CLI for intentisolates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llmintent.isolates import __version__
from llmintent.isolates._core.backends import available_backends, describe_backend
from llmintent.isolates._core.identify import identify_isolates
from llmintent.isolates._core.motifs import form_motifs
from llmintent.isolates._core.report import build_report, report_to_json, report_to_markdown
from llmintent.isolates._core.trajectory import trajectory_from_motifs
from llmintent.isolates._core.typology import classify_typology


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmintent-isolates",
        description="Identify isolates, classify typology, form motifs, map trajectories",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_input_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--text", type=str, default=None, help="Input text")
        p.add_argument("--features", type=str, default=None, help="Comma-separated floats or JSON object")
        p.add_argument("--graph", type=str, default=None, help="JSON graph {nodes,edges} or path to JSON file")
        p.add_argument("--backend", type=str, default="rule", choices=["rule", "hf", "llmintent", "soft"])
        p.add_argument("-o", "--output", type=str, default=None, help="Write JSON to path")

    p_id = sub.add_parser("identify", help="Identify isolates")
    add_input_args(p_id)

    p_ty = sub.add_parser("typology", help="Identify + classify typology")
    add_input_args(p_ty)

    p_mo = sub.add_parser("motifs", help="Form motifs from isolates")
    add_input_args(p_mo)

    p_tr = sub.add_parser("trajectory", help="Build reasoning trajectory")
    add_input_args(p_tr)

    p_re = sub.add_parser("report", help="Full isolate report")
    add_input_args(p_re)
    p_re.add_argument("--motifs", action="store_true", help="Include motifs")
    p_re.add_argument("--trajectory", action="store_true", help="Include trajectory")
    p_re.add_argument("--markdown", action="store_true", help="Also emit markdown (stdout or .md beside -o)")
    p_re.add_argument("--format", choices=["json", "markdown", "both"], default="json")

    p_be = sub.add_parser("backends", help="List available backends")

    p_ca = sub.add_parser(
        "causal",
        help="Layer motifs → indication vs IV causation report (AutoCausal/causaliv bridge)",
    )
    add_input_args(p_ca)
    p_ca.add_argument(
        "--outcome-hint",
        type=str,
        default=None,
        dest="outcome_hint",
        help="Hint for deriving outcome Y (e.g. decision, submit, on time)",
    )
    p_ca.add_argument("--n-bootstrap", type=int, default=48, dest="n_bootstrap")
    p_ca.add_argument("--seed", type=int, default=17)
    p_ca.add_argument(
        "--mock-iv",
        action="store_true",
        dest="mock_iv",
        help="Use mock IV estimator (tests / no causaliv)",
    )
    p_ca.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        dest="fmt",
    )
    p_ca.add_argument("--markdown", action="store_true", help="Force markdown (alias)")

    args = parser.parse_args(argv)

    if args.cmd == "backends":
        print(json.dumps(describe_backend(), indent=2))
        return 0

    text, features, graph = _parse_inputs(args)
    if text is None and features is None and graph is None:
        print("Provide --text and/or --features and/or --graph", file=sys.stderr)
        return 2

    if args.cmd in ("identify", "typology"):
        isos = identify_isolates(text=text, features=features, graph=graph, backend=args.backend)
        if args.cmd == "typology":
            isos = [classify_typology(i) for i in isos]
        payload = [i.to_dict() for i in isos]
        return _emit(payload, args.output)

    if args.cmd == "motifs":
        isos = identify_isolates(text=text, features=features, graph=graph, backend=args.backend)
        motifs = form_motifs(isos)
        payload = {
            "isolates": [i.to_dict() for i in isos],
            "motifs": [m.to_dict() for m in motifs],
            "backends": available_backends(),
        }
        return _emit(payload, args.output)

    if args.cmd == "trajectory":
        isos = identify_isolates(text=text, features=features, graph=graph, backend=args.backend)
        motifs = form_motifs(isos)
        traj = trajectory_from_motifs(motifs, isos)
        payload = {
            "isolates": [i.to_dict() for i in isos],
            "motifs": [m.to_dict() for m in motifs],
            "trajectory": traj.to_dict(),
        }
        return _emit(payload, args.output)

    if args.cmd == "report":
        # Default: include both. If either flag is set, honor the combination
        # (--trajectory implies motifs are computed for the path).
        if args.motifs or args.trajectory:
            include_motifs = True
            include_traj = bool(args.trajectory)
            if args.motifs and not args.trajectory:
                include_traj = False
            if args.trajectory:
                include_motifs = True
        else:
            include_motifs = True
            include_traj = True

        report = build_report(
            text=text,
            features=features,
            graph=graph,
            include_motifs=include_motifs,
            include_trajectory=include_traj,
            backend=args.backend,
        )
        fmt = args.format
        if args.markdown:
            fmt = "both" if args.output else "markdown"

        if fmt in ("json", "both"):
            report_to_json(report, args.output)
            if fmt == "json" and not args.output:
                print(report_to_json(report))
            elif fmt == "json" and args.output:
                print(f"Wrote {args.output}")
        if fmt in ("markdown", "both"):
            md_path = None
            if args.output and fmt == "both":
                md_path = str(Path(args.output).with_suffix(".md"))
            elif args.output and fmt == "markdown":
                md_path = args.output
            md = report_to_markdown(report, md_path)
            if not md_path:
                print(md)
            elif fmt == "both":
                print(f"Wrote {md_path}")
        return 0

    if args.cmd == "causal":
        from llmintent.isolates._core.causal import LayerCausalSuite

        suite = LayerCausalSuite(
            text=text,
            features=features,
            graph=graph,
            backend=args.backend,
        )
        result = suite.run(
            outcome_hint=args.outcome_hint,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            mock_iv=bool(args.mock_iv),
        )
        fmt = args.fmt
        if args.markdown:
            fmt = "both" if args.output else "markdown"

        if fmt in ("json", "both"):
            text_out = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
            if args.output:
                Path(args.output).write_text(text_out, encoding="utf-8")
                print(f"Wrote {args.output}")
            elif fmt == "json":
                print(text_out)
        if fmt in ("markdown", "both"):
            md = result.to_markdown()
            md_path = None
            if args.output and fmt == "both":
                md_path = str(Path(args.output).with_suffix(".md"))
            elif args.output and fmt == "markdown":
                md_path = args.output
            if md_path:
                Path(md_path).write_text(md, encoding="utf-8")
                print(f"Wrote {md_path}")
            else:
                print(md)
        return 0

    parser.error(f"Unknown command: {args.cmd}")
    return 2


def _parse_inputs(args: argparse.Namespace):
    text = args.text
    features = None
    graph = None
    if args.features:
        raw = args.features.strip()
        if raw.startswith("{") or raw.startswith("["):
            features = json.loads(raw)
        else:
            features = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if args.graph:
        raw = args.graph.strip()
        path = Path(raw)
        if path.is_file():
            graph = json.loads(path.read_text(encoding="utf-8"))
        else:
            graph = json.loads(raw)
    return text, features, graph


def _emit(payload, output: str | None) -> int:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(f"Wrote {output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
