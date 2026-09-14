"""Stdio MCP so any agent can drive the LLMIntent suite.

  python -m llmintent.mcp
  llmintent mcp
"""

from __future__ import annotations

import argparse
from typing import Any

from llmintent.mcp.install import host_configs
from llmintent.mcp.protocol import run_mcp_loop

INSTRUCTIONS = """LLMIntent MCP — fly-connectome anatomy, isolates, latent inspect.

Typical loop:
1. li_compile — closed region catalogue for a prompt
2. li_anatomy — what each region does and how occupancy varies through spans
3. li_draft — SLM/template (or your agent) writes the prose report
4. Optional: li_isolates, li_motifs, li_iv, li_latent (rule backend stays offline)

Do not claim fly neuropils inside the LLM. Residuals are correlates.
"""

_TEXT = {"type": "string", "description": "Prompt or English to map"}


MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "li_status",
        "description": "Suite status: version, anatomy regions, MCP tools.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "li_install",
        "description": "MCP host JSON for Cursor / VS Code / Claude / generic.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "li_models",
        "description": "List curated suite models (offline registry).",
        "inputSchema": {
            "type": "object",
            "properties": {"family": {"type": "string"}},
        },
    },
    {
        "name": "li_compile",
        "description": "Compile English onto fly-atlas regions. Unmatched text is dropped.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": _TEXT},
            "required": ["text"],
        },
    },
    {
        "name": "li_anatomy",
        "description": (
            "Map Anatomy: per-layer intent responsibility, complete graph, "
            "span occupancy, IV, A vs B. Offline unless a weighted model is loaded in-process."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": _TEXT,
                "draft": {"type": "boolean", "description": "Include template/SLM prose draft"},
                "slm": {"type": "string", "description": "Live SLM key (gpt2, qwen-0.5b) for the draft"},
                "endpoint": {"type": "string", "description": "OpenAI-compatible chat URL for the draft"},
                "region_a": {"type": "string", "default": "vision"},
                "region_b": {"type": "string", "default": "auditory"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "li_draft",
        "description": (
            "Draft a prose anatomy report. Uses a template unless slm or endpoint "
            "is set (or LLMINTENT_GUIDE_URL). Pass your own agent only via Python API."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": _TEXT,
                "slm": {"type": "string"},
                "endpoint": {"type": "string"},
                "endpoint_model": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "li_isolates",
        "description": "Identify intent isolates (offline rule backend).",
        "inputSchema": {
            "type": "object",
            "properties": {"text": _TEXT},
            "required": ["text"],
        },
    },
    {
        "name": "li_motifs",
        "description": "Form layer motifs from isolates.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": _TEXT},
            "required": ["text"],
        },
    },
    {
        "name": "li_trajectory",
        "description": (
            "Anatomy trajectory: all layers × all intents, negative-intent loci, "
            "MisAlign Flag. Set isolates=true for the motif path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": _TEXT,
                "isolates": {"type": "boolean"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "li_iv",
        "description": "Indication vs IV causation on layer motifs (mock IV by default).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": _TEXT,
                "outcome_hint": {"type": "string"},
                "mock_iv": {"type": "boolean", "default": True},
            },
            "required": ["text"],
        },
    },
    {
        "name": "li_latent",
        "description": "Latent ThoughtReport. Default backend=rule (offline). hf loads a model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": _TEXT,
                "backend": {"type": "string", "enum": ["rule", "hf"], "default": "rule"},
                "model": {"type": "string"},
            },
            "required": ["text"],
        },
    },
]


def _status() -> dict:
    from llmintent import __version__
    from llmintent.anatomy import region_ids

    return {
        "name": "llmintent",
        "version": __version__,
        "regions": list(region_ids()),
        "tools": [t["name"] for t in MCP_TOOLS],
        "note": "Fly atlas is an IV prior. Residuals are correlates.",
    }


def dispatch(name: str, args: dict[str, Any] | None = None) -> Any:
    args = args or {}
    text = str(args.get("text") or "")

    if name == "li_status":
        return _status()
    if name == "li_install":
        return host_configs()
    if name == "li_models":
        from llmintent.suite import list_models

        fam = args.get("family")
        return list_models(family=str(fam) if fam else None)
    if name == "li_compile":
        from llmintent.anatomy import compile_regions

        return compile_regions(text).to_dict()
    if name == "li_anatomy":
        from llmintent.anatomy import map_anatomy

        report = map_anatomy(
            text,
            region_a=str(args.get("region_a") or "vision"),
            region_b=str(args.get("region_b") or "auditory"),
            draft=bool(args.get("draft")),
            slm=args.get("slm"),
            endpoint=args.get("endpoint"),
            mock_iv=True,
        )
        return report.to_dict()
    if name == "li_draft":
        from llmintent.anatomy import map_anatomy
        from llmintent.anatomy.guide import draft_anatomy_report

        report = map_anatomy(text, mock_iv=True, ablate=True)
        draft = draft_anatomy_report(
            report,
            slm=args.get("slm"),
            endpoint=args.get("endpoint"),
            endpoint_model=args.get("endpoint_model"),
        )
        return draft.to_dict()
    if name == "li_isolates":
        from llmintent.isolates import identify_isolates

        return [i.to_dict() for i in identify_isolates(text=text)]
    if name == "li_motifs":
        from llmintent.isolates import form_motifs, identify_isolates

        isos = identify_isolates(text=text)
        return {
            "isolates": [i.to_dict() for i in isos],
            "motifs": [m.to_dict() for m in form_motifs(isos)],
        }
    if name == "li_trajectory":
        if args.get("isolates"):
            from llmintent.isolates import form_motifs, identify_isolates, trajectory_from_motifs

            isos = identify_isolates(text=text)
            motifs = form_motifs(isos)
            traj = trajectory_from_motifs(motifs, isos)
            return traj.to_dict()
        from llmintent.anatomy import trajectory as anatomy_trajectory

        traj = anatomy_trajectory(text, print_flag=False, all_layers=True)
        payload = traj.to_dict()
        if traj.misalign and traj.misalign.triggered:
            payload["misalign_banner"] = traj.misalign.banner()
        return payload
    if name == "li_iv":
        from llmintent.iv_motifs import LayerCausalSuite

        suite = LayerCausalSuite(text=text)
        mock = args.get("mock_iv")
        result = suite.run(
            outcome_hint=args.get("outcome_hint"),
            mock_iv=True if mock is None else bool(mock),
            n_bootstrap=24,
            seed=3,
        )
        return result.to_dict()
    if name == "li_latent":
        from llmintent import latent as li_latent

        backend = str(args.get("backend") or "rule")
        report = li_latent.inspect_text(
            text,
            backend=backend,
            model=args.get("model"),
            include_sae=False,
            include_probe_train=False,
            require_hf=backend == "hf",
        )
        return report.to_dict() if hasattr(report, "to_dict") else report
    return {"error": f"unknown tool: {name}", "tools": [t["name"] for t in MCP_TOOLS]}


def list_resources() -> list[dict[str, Any]]:
    return [
        {
            "uri": "anatomy://atlas",
            "name": "Fly-connectome atlas",
            "mimeType": "application/json",
        }
    ]


def read_resource(uri: str) -> dict[str, Any]:
    from llmintent.anatomy import default_atlas, what_region_does
    from llmintent.anatomy.atlas import REGIONS

    if uri == "anatomy://atlas":
        atlas = default_atlas()
        payload = atlas.to_dict()
        payload["does"] = {r.id: what_region_does(r.id) for r in REGIONS}
        text = __import__("json").dumps(payload, indent=2)
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": text}]}
    return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": f"unknown resource {uri}"}]}


def list_prompts() -> list[dict[str, Any]]:
    return [
        {
            "name": "anatomy_report",
            "description": "Map a prompt onto fly-atlas regions and draft a report",
            "arguments": [{"name": "text", "required": True}],
        }
    ]


def get_prompt(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    text = str(arguments.get("text") or "I hear a song because a dark shape is looming.")
    return {
        "description": name,
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Call li_anatomy then li_draft on: {text}\n"
                        "Report what each region does and how it varies through the prompt."
                    ),
                },
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLMIntent MCP server (stdio)")
    parser.add_argument("--install", action="store_true", help="Print host MCP JSON and exit")
    parser.add_argument("--jsonl", action="store_true", help="Accepted for host compatibility")
    args = parser.parse_args(argv)
    if args.install:
        import json

        print(json.dumps(host_configs(), indent=2))
        return 0
    from llmintent import __version__

    return run_mcp_loop(
        server_name="llmintent",
        server_version=__version__,
        list_tools=lambda: MCP_TOOLS,
        call_tool=dispatch,
        list_resources=list_resources,
        read_resource=read_resource,
        list_prompts=list_prompts,
        get_prompt=get_prompt,
        instructions=INSTRUCTIONS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
