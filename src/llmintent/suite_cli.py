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

    latent = sub.add_parser(
        "latent",
        help="Latent thought inspection (ThoughtReport; correlates/probes, not mind-reading)",
    )
    latent.add_argument("--text", type=str, required=True, help="Input text")
    latent.add_argument(
        "--backend",
        type=str,
        default="rule",
        choices=["rule", "hf"],
        help="rule=offline vendored; hf=requires latentintent[hf] or torch stack",
    )
    latent.add_argument("--model", type=str, default=None, help="HF model id or suite key (qwen:27b)")
    latent.add_argument(
        "--family",
        type=str,
        default=None,
        choices=["qwen", "mistral", "minimax", "glm", "legacy"],
        help="Suite family (soft-resolved when llmintent suite / latentintent present)",
    )
    latent.add_argument(
        "--size",
        type=str,
        default=None,
        help="Suite size (tiny/small/medium/large/xl, or 27b for Qwen)",
    )
    latent.add_argument(
        "--4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load NF4 (required for 27B on ~24 GB GPUs)",
    )
    latent.add_argument("--layer-stride", type=int, default=4, dest="layer_stride")
    latent.add_argument("--no-sae", action="store_true")
    latent.add_argument("--no-probe", action="store_true")
    latent.add_argument("-o", "--output", type=str, default=None)
    latent.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        dest="fmt",
    )
    latent.add_argument(
        "--status",
        action="store_true",
        help="Print latent backend describe() JSON instead of inspecting",
    )

    compile_p = sub.add_parser(
        "compile",
        help="Compile English onto closed LLM-region catalogue (fly intent docs)",
    )
    compile_p.add_argument("--text", required=True)
    compile_p.add_argument("-o", "--output", default=None)
    compile_p.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        dest="fmt",
    )

    anatomy = sub.add_parser(
        "anatomy",
        help="Map LLM anatomy: what each region does, occupancy through each span, IV, A vs B ablation",
    )
    anatomy.add_argument("--text", required=True)
    anatomy.add_argument("--region-a", default="vision", dest="region_a")
    anatomy.add_argument("--region-b", default="auditory", dest="region_b")
    anatomy.add_argument("--model", default=None, help="HF id or suite key; omit for offline planted ablation")
    anatomy.add_argument(
        "--4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load NF4 (needed for 27B on ~24 GB GPUs)",
    )
    anatomy.add_argument("--no-ablate", action="store_true")
    anatomy.add_argument("--weights", action="store_true", help="Also SVD-map FFN weights (needs --model)")
    anatomy.add_argument("--seed", type=int, default=17)
    anatomy.add_argument("-o", "--output", default=None)
    anatomy.add_argument("--draft", action="store_true", help="Append a guided prose report (template, or --slm / --endpoint)")
    anatomy.add_argument("--slm", default=None, help="Live SLM key to draft the report (gpt2, qwen-0.5b, …)")
    anatomy.add_argument("--endpoint", default=None, help="OpenAI-compatible chat URL for the draft")
    anatomy.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        dest="fmt",
    )

    mcp = sub.add_parser("mcp", help="Stdio MCP server so agents can drive the suite")
    mcp.add_argument("--install", action="store_true", help="Print host MCP JSON and exit")

    guide = sub.add_parser(
        "guide",
        help="Draft a prose anatomy report (template, --slm, or --endpoint)",
    )
    guide.add_argument("--text", required=True)
    guide.add_argument("--slm", default=None, help="Live SLM key (gpt2, qwen-0.5b, …)")
    guide.add_argument("--endpoint", default=None, help="OpenAI-compatible chat URL")
    guide.add_argument("--endpoint-model", default=None, dest="endpoint_model")
    guide.add_argument("-o", "--output", default=None)
    guide.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        dest="fmt",
    )


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
    if cmd == "latent":
        return _cmd_latent(args)
    if cmd == "compile":
        return _cmd_compile(args)
    if cmd == "anatomy":
        return _cmd_anatomy(args)
    if cmd == "mcp":
        from llmintent.mcp.server import main as mcp_main

        flags: list[str] = []
        if getattr(args, "install", False):
            flags.append("--install")
        return mcp_main(flags)
    if cmd == "guide":
        return _cmd_guide(args)
    if cmd == "trajectory" and getattr(args, "text", None):
        # Dual-mode: trajectory --text → motif reasoning path
        return _cmd_reasoning_trajectory(args)
    return None


def _cmd_latent(args: argparse.Namespace) -> int:
    from llmintent import latent as li_latent

    if getattr(args, "status", False):
        print(json.dumps(li_latent.describe(), indent=2))
        return 0

    report = li_latent.inspect_text(
        args.text,
        backend=getattr(args, "backend", "rule") or "rule",
        model=getattr(args, "model", None),
        family=getattr(args, "family", None),
        size=getattr(args, "size", None),
        include_sae=not getattr(args, "no_sae", False),
        include_probe_train=not getattr(args, "no_probe", False),
        load_in_4bit=bool(getattr(args, "load_in_4bit", False)),
        layer_stride=int(getattr(args, "layer_stride", 4) or 4),
        require_hf=getattr(args, "backend", "rule") == "hf",
    )
    if getattr(args, "fmt", "json") == "markdown" and hasattr(report, "summary_lines"):
        md_bits = ["# Latent thoughts", ""]
        meta = getattr(report, "metadata", {}) or {}
        if getattr(report, "model_name", None):
            md_bits.append(f"**Model:** `{report.model_name}`")
        md_bits.append(f"**Prompt:** {args.text}")
        compiled = meta.get("compiled_regions")
        if compiled:
            md_bits.append(f"**Compile prior:** {', '.join(compiled)}")
        md_bits.extend(["", "## Layer logit lens", ""])
        for row in getattr(report, "logit_lens", []) or []:
            toks = ", ".join(
                (t.get("token") if isinstance(t, dict) else str(t))
                for t in (row.get("top_tokens") or [])[:4]
            )
            md_bits.append(
                f"- L{row.get('layer')}: `{row.get('region', '')}` — {toks}"
            )
        occ = meta.get("occupancy") or {}
        if occ:
            md_bits.extend(["", "## Occupancy", ""])
            for rid, v in sorted(occ.items(), key=lambda kv: -float(kv[1])):
                md_bits.append(f"- `{rid}`: {float(v):.3f}")
        md_bits.extend(["", "## Caveats"])
        for c in getattr(report, "caveats", [])[:6]:
            md_bits.append(f"- {c}")
        text = "\n".join(md_bits) + "\n"
        out = getattr(args, "output", None)
        if out:
            Path(out).write_text(text, encoding="utf-8")
            print(f"Wrote {out}")
        else:
            _safe_print(text)
        return 0
    payload = report.to_dict() if hasattr(report, "to_dict") else report
    text = json.dumps(payload, indent=2)
    out = getattr(args, "output", None)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    from llmintent.anatomy import compile_regions

    plan = compile_regions(args.text)
    if getattr(args, "fmt", "json") == "markdown":
        lines = ["# Region compile", "", f"**Text:** {plan.text}", ""]
        for h in plan.hits:
            lines.append(f"- `{h.region}` ({h.via}, {h.score:.3f}): {h.handles}")
        if plan.dropped:
            lines.append("")
            lines.append("Dropped: " + "; ".join(plan.dropped))
        _safe_print("\n".join(lines))
        return 0
    return _emit(plan.to_dict(), getattr(args, "output", None))


def _cmd_anatomy(args: argparse.Namespace) -> int:
    from llmintent.anatomy import map_anatomy

    bundle = None
    model = getattr(args, "model", None)
    if model:
        from llmintent.models import load_model_bundle
        from llmintent.suite import resolve_model_spec

        spec = resolve_model_spec(model=model, use_env=False)
        hf_id = spec.hf_id if spec is not None else model
        fourbit = bool(getattr(args, "load_in_4bit", False))
        if spec is not None and spec.size == "27b":
            fourbit = True
        bundle = load_model_bundle(hf_id, load_in_4bit=fourbit)
    report = map_anatomy(
        args.text,
        bundle=bundle,
        region_a=args.region_a,
        region_b=args.region_b,
        ablate=not getattr(args, "no_ablate", False),
        include_weights=bool(getattr(args, "weights", False)),
        seed=int(getattr(args, "seed", 17)),
        mock_iv=bundle is None,
        draft=bool(getattr(args, "draft", False) or getattr(args, "slm", None) or getattr(args, "endpoint", None)),
        slm=getattr(args, "slm", None),
        endpoint=getattr(args, "endpoint", None),
    )
    if getattr(args, "fmt", "markdown") == "json":
        return _emit(report.to_dict(), getattr(args, "output", None))
    _safe_print(report.to_markdown())
    return 0


def _cmd_guide(args: argparse.Namespace) -> int:
    from llmintent.anatomy import map_anatomy
    from llmintent.anatomy.guide import draft_anatomy_report

    report = map_anatomy(args.text, mock_iv=True)
    draft = draft_anatomy_report(
        report,
        slm=getattr(args, "slm", None),
        endpoint=getattr(args, "endpoint", None),
        endpoint_model=getattr(args, "endpoint_model", None),
    )
    if getattr(args, "fmt", "markdown") == "json":
        return _emit(draft.to_dict(), getattr(args, "output", None))
    _safe_print(draft.markdown)
    return 0


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
