"""Umbrella CLI helpers for isolates / motifs / IV suite commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def add_suite_parsers(sub: argparse._SubParsersAction) -> None:
    """Register isolates, motifs, reasoning-trajectory, iv-motifs under llmintent."""

    isolates = sub.add_parser(
        "isolates",
        help="Identify isolates + typology (suite; offline rule backend)",
    )
    isolates_sub = isolates.add_subparsers(dest="isolates_cmd")
    _add_input_args(isolates)  # default: identify when no subcommand
    for name, help_text in (
        ("identify", "Identify isolates"),
        ("typology", "Identify + classify typology"),
        ("report", "Full isolate report"),
        ("backends", "List available backends"),
    ):
        p = isolates_sub.add_parser(name, help=help_text)
        if name != "backends":
            _add_input_args(p)
        if name == "report":
            p.add_argument("--motifs", action="store_true")
            p.add_argument("--trajectory", action="store_true")
            p.add_argument("--format", choices=["json", "markdown", "both"], default="json")
            p.add_argument("--markdown", action="store_true")

    motifs = sub.add_parser("motifs", help="Form layer motifs from isolates")
    _add_input_args(motifs)

    # reasoning-trajectory: explicit suite name (activation trajectory keeps `trajectory --prompt`)
    rtraj = sub.add_parser(
        "reasoning-trajectory",
        help="Reasoning trajectory from layer motifs (isolates suite)",
    )
    _add_input_args(rtraj)

    iv = sub.add_parser(
        "iv-motifs",
        help="Layer motifs → indication vs IV causation (AutoCausal/causaliv soft)",
    )
    _add_input_args(iv)
    iv.add_argument("--outcome-hint", default=None, dest="outcome_hint")
    iv.add_argument("--n-bootstrap", type=int, default=48, dest="n_bootstrap")
    iv.add_argument("--seed", type=int, default=17)
    iv.add_argument("--mock-iv", action="store_true", dest="mock_iv")
    iv.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        dest="fmt",
    )
    iv.add_argument("--markdown", action="store_true")

    # Also expose causal-layers alias
    cl = sub.add_parser(
        "causal-layers",
        help="Alias of iv-motifs (indication vs causation)",
    )
    _add_input_args(cl)
    cl.add_argument("--outcome-hint", default=None, dest="outcome_hint")
    cl.add_argument("--n-bootstrap", type=int, default=48, dest="n_bootstrap")
    cl.add_argument("--seed", type=int, default=17)
    cl.add_argument("--mock-iv", action="store_true", dest="mock_iv")
    cl.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        dest="fmt",
    )
    cl.add_argument("--markdown", action="store_true")


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--text", type=str, default=None, help="Input text")
    p.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated floats or JSON object",
    )
    p.add_argument(
        "--graph",
        type=str,
        default=None,
        help="JSON graph {nodes,edges} or path to JSON file",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="rule",
        choices=["rule", "hf", "llmintent", "soft"],
    )
    p.add_argument("-o", "--output", type=str, default=None, help="Write JSON to path")


def handle_suite_command(args: argparse.Namespace) -> int | None:
    """Dispatch suite commands. Returns exit code, or None if not a suite cmd."""
    cmd = args.command
    if cmd == "isolates":
        return _cmd_isolates(args)
    if cmd == "motifs":
        return _cmd_motifs(args)
    if cmd == "reasoning-trajectory":
        return _cmd_reasoning_trajectory(args)
    if cmd in ("iv-motifs", "causal-layers"):
        return _cmd_iv_motifs(args)
    if cmd == "trajectory" and getattr(args, "text", None):
        # Dual-mode: trajectory --text → motif reasoning path
        return _cmd_reasoning_trajectory(args)
    return None


def maybe_patch_trajectory_parser(trajectory_parser: argparse.ArgumentParser) -> None:
    """Allow activation trajectory to also accept --text for suite mode."""
    trajectory_parser.add_argument(
        "--text",
        default=None,
        help="If set (without requiring a model), build isolates reasoning trajectory",
    )


def _parse_inputs(args: argparse.Namespace):
    text = getattr(args, "text", None)
    features = None
    graph = None
    if getattr(args, "features", None):
        raw = args.features.strip()
        if raw.startswith("{") or raw.startswith("["):
            features = json.loads(raw)
        else:
            features = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if getattr(args, "graph", None):
        raw = args.graph.strip()
        path = Path(raw)
        if path.is_file():
            graph = json.loads(path.read_text(encoding="utf-8"))
        else:
            graph = json.loads(raw)
    return text, features, graph


def _safe_print(text: str) -> None:
    """Print text without crashing on Windows charmap consoles."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))


def _emit(payload, output: str | None) -> int:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(f"Wrote {output}")
    else:
        _safe_print(text)
    return 0


def _cmd_isolates(args: argparse.Namespace) -> int:
    from llmintent.isolates import (
        build_report,
        classify_typology,
        identify_isolates,
        report_to_json,
        report_to_markdown,
    )
    from llmintent.isolates._core.backends import describe_backend

    sub = getattr(args, "isolates_cmd", None) or "identify"
    if sub == "backends":
        print(json.dumps(describe_backend(), indent=2))
        return 0

    text, features, graph = _parse_inputs(args)
    if text is None and features is None and graph is None:
        print("Provide --text and/or --features and/or --graph", file=sys.stderr)
        return 2

    backend = getattr(args, "backend", "rule")
    if sub in ("identify", "typology"):
        isos = identify_isolates(text=text, features=features, graph=graph, backend=backend)
        if sub == "typology":
            isos = [classify_typology(i) for i in isos]
        return _emit([i.to_dict() for i in isos], getattr(args, "output", None))

    if sub == "report":
        include_motifs = True
        include_traj = True
        if getattr(args, "motifs", False) or getattr(args, "trajectory", False):
            include_motifs = True
            include_traj = bool(getattr(args, "trajectory", False))
            if getattr(args, "motifs", False) and not getattr(args, "trajectory", False):
                include_traj = False
        report = build_report(
            text=text,
            features=features,
            graph=graph,
            include_motifs=include_motifs,
            include_trajectory=include_traj,
            backend=backend,
        )
        fmt = getattr(args, "format", "json")
        if getattr(args, "markdown", False):
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

    return 2


def _cmd_motifs(args: argparse.Namespace) -> int:
    from llmintent.isolates import available_backends, form_motifs, identify_isolates

    text, features, graph = _parse_inputs(args)
    if text is None and features is None and graph is None:
        print("Provide --text and/or --features and/or --graph", file=sys.stderr)
        return 2
    isos = identify_isolates(
        text=text, features=features, graph=graph, backend=getattr(args, "backend", "rule")
    )
    motifs = form_motifs(isos)
    return _emit(
        {
            "isolates": [i.to_dict() for i in isos],
            "motifs": [m.to_dict() for m in motifs],
            "backends": available_backends(),
        },
        getattr(args, "output", None),
    )


def _cmd_reasoning_trajectory(args: argparse.Namespace) -> int:
    from llmintent.isolates import form_motifs, identify_isolates, trajectory_from_motifs

    text, features, graph = _parse_inputs(args)
    if text is None and features is None and graph is None:
        print("Provide --text and/or --features and/or --graph", file=sys.stderr)
        return 2
    isos = identify_isolates(
        text=text, features=features, graph=graph, backend=getattr(args, "backend", "rule")
    )
    motifs = form_motifs(isos)
    traj = trajectory_from_motifs(motifs, isos)
    return _emit(
        {
            "isolates": [i.to_dict() for i in isos],
            "motifs": [m.to_dict() for m in motifs],
            "trajectory": traj.to_dict(),
        },
        getattr(args, "output", None),
    )


def _cmd_iv_motifs(args: argparse.Namespace) -> int:
    from llmintent.iv_motifs import LayerCausalSuite

    text, features, graph = _parse_inputs(args)
    if text is None and features is None and graph is None:
        print("Provide --text and/or --features and/or --graph", file=sys.stderr)
        return 2
    suite = LayerCausalSuite(
        text=text,
        features=features,
        graph=graph,
        backend=getattr(args, "backend", "rule"),
    )
    result = suite.run(
        outcome_hint=getattr(args, "outcome_hint", None),
        n_bootstrap=getattr(args, "n_bootstrap", 48),
        seed=getattr(args, "seed", 17),
        mock_iv=bool(getattr(args, "mock_iv", False)),
    )
    fmt = getattr(args, "fmt", "markdown")
    if getattr(args, "markdown", False):
        fmt = "both" if args.output else "markdown"
    if fmt in ("json", "both"):
        text_out = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(text_out, encoding="utf-8")
            print(f"Wrote {args.output}")
        elif fmt == "json":
            _safe_print(text_out)
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
            _safe_print(md)
    return 0


def run_isolates_argv(argv: Sequence[str] | None = None) -> int:
    """Forward to vendored isolates CLI (``python -m llmintent.isolates``)."""
    from llmintent.isolates._core.cli import main as isolates_main

    return isolates_main(list(argv) if argv is not None else None)
