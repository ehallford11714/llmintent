"""SLM / agent guide: draft a prose anatomy report from structured facts.

Default is a deterministic template (offline). Pass ``agent`` (callable) or
an OpenAI-compatible ``endpoint`` / live ``slm`` key if the user provides one.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from llmintent.anatomy.report import AnatomyReport

SYSTEM = (
    "You draft LLMIntent anatomy reports. Use only the JSON facts given. "
    "Do not invent neuropils, papers, or occupancy. Say when a region is silent. "
    "Fly connectome is an IV prior, not a claim the model is a fly. "
    "Residual tokens are correlates, not mind-reading."
)

AgentFn = Callable[[str, str], str]


@dataclass
class GuideDraft:
    markdown: str
    backend: str
    model: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "markdown": self.markdown,
            "backend": self.backend,
            "model": self.model,
            "notes": list(self.notes),
        }


def facts_for_guide(report: AnatomyReport) -> dict[str, Any]:
    """Compact JSON an SLM or agent can consume without the full atlas."""
    trace = report.trace
    active = [
        {
            "id": t.id,
            "does": t.does,
            "band": t.band,
            "peak_span": t.peak_span,
            "peak_text": t.peak_text,
            "varies": t.varies,
            "variation": round(t.variation, 4),
        }
        for t in (trace.regions if trace else [])
        if t.peak_span is not None
    ]
    silent = [t.id for t in (trace.regions if trace else []) if t.peak_span is None]
    spans = [s.to_dict() for s in (trace.spans if trace else [])]
    ablation = report.ablation.to_dict() if report.ablation else None
    return {
        "prompt": report.text,
        "model": report.model_name or "offline",
        "compiled": list(report.plan.regions),
        "spans": spans,
        "active_regions": active,
        "silent_regions": silent,
        "ablation": ablation,
        "caveats": [
            "Fly connectome is an identification prior, not a fly-brained LLM.",
            "Compile drops unmatched English.",
            "Ablation is a next-token or linear-readout shift.",
        ],
    }


def template_draft(report: AnatomyReport) -> str:
    """Deterministic prose from the structured map (no model required)."""
    facts = facts_for_guide(report)
    lines = [
        "# Anatomy report",
        "",
        f"**Prompt:** {facts['prompt']}",
        f"**Model:** `{facts['model']}`",
        f"**Compiled:** {', '.join(facts['compiled']) or '(none)'}",
        "",
        "## What each active region does, and how it varies through the prompt",
        "",
    ]
    if facts["spans"]:
        lines.append("Spans:")
        for s in facts["spans"]:
            regs = ", ".join(s["regions"]) or "—"
            lines.append(f"- [{s['index']}] {s['text']} → {regs}")
        lines.append("")
    for r in facts["active_regions"]:
        lines.append(f"### `{r['id']}` ({r['band']})")
        lines.append("")
        lines.append(r["does"])
        lines.append("")
        lines.append(r["varies"])
        lines.append("")
    if facts["silent_regions"]:
        lines.append("Silent on this prompt: " + ", ".join(f"`{x}`" for x in facts["silent_regions"]))
        lines.append("")
    if facts["ablation"]:
        a = facts["ablation"]
        lines.append("## Ablation")
        lines.append("")
        lines.append(
            f"Activating `{a.get('region_a')}` vs `{a.get('region_b')}` "
            f"{'changed' if a.get('changed') else 'did not change'} the readout "
            f"(KL={a.get('kl_ab', 0):.3f})."
        )
        lines.append("")
    lines.append("## Caveats")
    for c in facts["caveats"]:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


def _call_agent(agent: AgentFn, facts: dict) -> str:
    user = (
        "Draft a short markdown anatomy report from these facts only:\n\n"
        + json.dumps(facts, indent=2)
    )
    return agent(SYSTEM, user)


def _call_endpoint(
    facts: dict,
    *,
    url: str,
    model: str,
    api_key: str | None,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "Draft a short markdown anatomy report from these facts only:\n\n"
                + json.dumps(facts, indent=2),
            },
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Guide endpoint returned no choices")
    return str(choices[0].get("message", {}).get("content") or "").strip()


def _call_slm(facts: dict, slm: str, *, max_new_tokens: int = 256) -> str:
    from llmintent.live.generate import generate_completion
    from llmintent.live.session import LiveSession, LiveSessionConfig

    session = LiveSession(LiveSessionConfig(model_key=slm, retracement_mode="baseline"))
    session.load()
    prompt = (
        SYSTEM
        + "\n\nFacts:\n"
        + json.dumps(facts, indent=2)
        + "\n\nMarkdown report:\n"
    )
    text, _ = generate_completion(
        session,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        retracement_mode="baseline",
    )
    return (text or "").strip()


def draft_anatomy_report(
    report: AnatomyReport,
    *,
    agent: AgentFn | None = None,
    slm: str | None = None,
    endpoint: str | None = None,
    endpoint_model: str | None = None,
    api_key: str | None = None,
) -> GuideDraft:
    """
    Draft prose for an AnatomyReport.

    Resolution: ``agent`` callable → OpenAI-compatible ``endpoint`` (or env
    ``LLMINTENT_GUIDE_URL``) → live ``slm`` key → deterministic template.
    """
    facts = facts_for_guide(report)
    template = template_draft(report)
    notes: list[str] = []

    if agent is not None:
        try:
            md = _call_agent(agent, facts)
            return GuideDraft(markdown=md or template, backend="agent", notes=notes)
        except Exception as exc:
            notes.append(f"agent_failed: {exc}")
            return GuideDraft(markdown=template, backend="template", notes=notes)

    url = endpoint or os.environ.get("LLMINTENT_GUIDE_URL")
    model = endpoint_model or os.environ.get("LLMINTENT_GUIDE_MODEL") or "local"
    key = api_key or os.environ.get("LLMINTENT_GUIDE_KEY") or os.environ.get("OPENAI_API_KEY")
    if url:
        try:
            md = _call_endpoint(facts, url=url, model=model, api_key=key)
            return GuideDraft(markdown=md or template, backend="endpoint", model=model, notes=notes)
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            notes.append(f"endpoint_failed: {exc}")

    if slm:
        try:
            md = _call_slm(facts, slm)
            return GuideDraft(markdown=md or template, backend="slm", model=slm, notes=notes)
        except Exception as exc:
            notes.append(f"slm_failed: {exc}")

    return GuideDraft(markdown=template, backend="template", notes=notes)
