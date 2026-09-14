"""Anatomy report: what each LLM region handles, how it is wired, ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmintent.anatomy.ablation import AblationResult, ablate_model, plant_and_ablate
from llmintent.anatomy.atlas import REGIONS, Atlas, what_region_does
from llmintent.anatomy.compile import RegionPlan
from llmintent.anatomy.connectome import default_atlas, literature_region_connectome
from llmintent.anatomy.iv_engine import AnatomyIVResult, iv_from_text
from llmintent.anatomy.svd_map import SVDAnatomy, map_activations, map_weights
from llmintent.anatomy.trace import PromptTrace, RegionTrace, trace_prompt


@dataclass
class RegionCard:
    id: str
    handles: str
    does: str
    fly_neuropil: str
    band: str
    integrates_with: list[str]
    occupancy: float
    compiled: bool
    depth: tuple[float, float]
    series: list[float] = field(default_factory=list)
    peak_span: int | None = None
    peak_text: str = ""
    variation: float = 0.0
    varies: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "handles": self.handles,
            "does": self.does,
            "fly_neuropil": self.fly_neuropil,
            "band": self.band,
            "integrates_with": list(self.integrates_with),
            "occupancy": round(self.occupancy, 4),
            "compiled": self.compiled,
            "depth": list(self.depth),
            "series": [round(x, 4) for x in self.series],
            "peak_span": self.peak_span,
            "peak_text": self.peak_text,
            "variation": round(self.variation, 4),
            "varies": self.varies,
        }


@dataclass
class AnatomyReport:
    text: str
    atlas: Atlas
    plan: RegionPlan
    iv: AnatomyIVResult
    svd: SVDAnatomy | None = None
    ablation: AblationResult | None = None
    cards: list[RegionCard] = field(default_factory=list)
    model_name: str | None = None
    notes: list[str] = field(default_factory=list)
    trace: PromptTrace | None = None
    draft: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model_name,
            "purpose": (
                "Map the anatomy of the LLM: what each region handles, "
                "how it is integrated, and whether activating A vs B changes output."
            ),
            "plan": self.plan.to_dict(),
            "regions": [c.to_dict() for c in self.cards],
            "trace": self.trace.to_dict() if self.trace else None,
            "draft": self.draft,
            "integration": self.atlas.to_dict()["integrates"],
            "iv": self.iv.to_dict(),
            "svd": self.svd.to_dict() if self.svd else None,
            "ablation": self.ablation.to_dict() if self.ablation else None,
            "notes": list(self.notes),
            "caveats": [
                "Fly connectome is an IV / depth-band prior, not a claim the LLM is a fly.",
                "Compile drops unmatched English; it does not invent regions.",
                "Ablation shows a next-token (or linear-readout) shift, not mind-reading.",
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# LLM anatomy map",
            "",
            f"**Prompt:** {self.text or '(none)'}  ",
            f"**Model:** `{self.model_name or 'offline'}`  ",
            f"**Compiled regions:** {', '.join(self.plan.regions) or '(none)'}  ",
            "",
            "## Regions",
            "",
            "| Region | Does | Band | Occupancy | Varies through prompt |",
            "|--------|------|------|-----------|-----------------------|",
        ]
        for c in self.cards:
            if not c.compiled and c.occupancy <= 0:
                continue
            job = c.does.split(" Band:")[0]
            varies = c.varies.split(". ")[0] if c.varies else "—"
            lines.append(
                f"| `{c.id}` | {job} | {c.band} | "
                f"{c.occupancy:.3f} | {varies} |"
            )
        silent = [c.id for c in self.cards if not c.compiled and c.occupancy <= 0]
        lines.append("")
        if self.trace and self.trace.spans:
            lines.append("## Through the prompt")
            lines.append("")
            for s in self.trace.spans:
                regs = ", ".join(f"`{r}`" for r in s.regions) or "—"
                lines.append(f"- **[{s.index}]** {s.text} → {regs}")
            lines.append("")
        if silent:
            lines.append("Silent regions: " + ", ".join(f"`{x}`" for x in silent))
            lines.append("")
        active = [c for c in self.cards if c.compiled or c.occupancy > 0]
        if active:
            lines.append("## What each active region does")
            lines.append("")
            for c in active:
                lines.append(f"### `{c.id}`")
                lines.append("")
                lines.append(c.does)
                lines.append("")
                if c.varies:
                    lines.append(c.varies)
                    lines.append("")
        if self.draft:
            lines.append("## Guided draft")
            lines.append("")
            lines.append(self.draft)
            lines.append("")
        if self.iv.causation_edges:
            lines.append("## Connectome-guided IV")
            lines.append("")
            lines.append("| X | Z (instrument) | beta_IV | F1 |")
            lines.append("|---|----------------|---------|----|")
            for e in self.iv.causation_edges[:8]:
                lines.append(
                    f"| `{e.source}` | `{e.instrument}` | {e.beta_iv:+.4f} | {e.first_stage_f:.1f} |"
                )
            lines.append("")
            if self.iv.exclusion_violations:
                lines.append("Exclusion violations (sensory short-pipes onto motor):")
                for v in self.iv.exclusion_violations:
                    lines.append(f"- `{v['source']}` → `{v['target']}` (w={v['weight']})")
                lines.append("")
        if self.ablation is not None:
            a = self.ablation
            lines.append("## Ablation (region A vs B)")
            lines.append("")
            lines.append(
                f"Activate **{a.region_a}** ({a.handles_a}) vs **{a.region_b}** "
                f"({a.handles_b})."
            )
            lines.append("")
            lines.append(f"- top A: `{a.top_a}`")
            lines.append(f"- top B: `{a.top_b}`")
            lines.append(f"- KL(A‖B) = {a.kl_ab:.4f}")
            lines.append(f"- output changed: **{'yes' if a.changed else 'no'}**")
            lines.append("")
        lines.append("## Caveats")
        for n in self.to_dict()["caveats"]:
            lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)


def _cards(
    atlas: Atlas,
    occupancy: dict[str, float],
    compiled: set[str],
    traces: dict[str, RegionTrace] | None = None,
) -> list[RegionCard]:
    out: list[RegionCard] = []
    for r in REGIONS:
        tr = (traces or {}).get(r.id)
        out.append(
            RegionCard(
                id=r.id,
                handles=r.handles,
                does=what_region_does(r.id),
                fly_neuropil=r.fly_neuropil,
                band=r.band,
                integrates_with=list(atlas.integrates.get(r.id, ())),
                occupancy=float(occupancy.get(r.id, 0.0)),
                compiled=r.id in compiled,
                depth=r.depth,
                series=list(tr.series) if tr else [],
                peak_span=tr.peak_span if tr else None,
                peak_text=tr.peak_text if tr else "",
                variation=tr.variation if tr else 0.0,
                varies=tr.varies if tr else "",
            )
        )
    return out


def map_anatomy(
    text: str,
    *,
    bundle: Any | None = None,
    region_a: str = "vision",
    region_b: str = "auditory",
    mock_iv: bool = True,
    include_weights: bool = False,
    ablate: bool = True,
    seed: int = 17,
    draft: bool = False,
    agent: Any | None = None,
    slm: str | None = None,
    endpoint: str | None = None,
) -> AnatomyReport:
    """
    Map LLM anatomy for ``text``.

    Offline (no bundle): compile + connectome IV + planted A vs B ablation.
    With a model bundle: add activation SVD occupancy and residual-stream ablation.
    """
    atlas = default_atlas()
    plan, iv = iv_from_text(text, mock_iv=mock_iv, seed=seed)
    occupancy = dict(plan.occupancy())
    trace = trace_prompt(text)
    svd: SVDAnatomy | None = None
    ablation: AblationResult | None = None
    model_name = None
    notes = list(plan.notes)

    if bundle is not None:
        model_name = getattr(bundle, "name", None)
        svd = map_activations(bundle, text)
        for rid, val in svd.occupancy.items():
            occupancy[rid] = max(occupancy.get(rid, 0.0), val)
        plan, iv = iv_from_text(text, occupancy=occupancy, mock_iv=mock_iv, seed=seed)
        if include_weights:
            wmap = map_weights(bundle)
            notes.extend(wmap.notes)
        if ablate:
            ablation = ablate_model(bundle, text, svd, region_a, region_b)
        notes.extend(svd.notes)
    elif ablate:
        ablation, _axes = plant_and_ablate(region_a, region_b, seed=seed)
        notes.append("Offline ablation uses planted orthogonal axes and a linear readout.")

    cards = _cards(atlas, occupancy, set(plan.regions), {t.id: t for t in trace.regions})
    notes.append(
        f"Connectome source: {literature_region_connectome().source}."
    )
    report = AnatomyReport(
        text=text,
        atlas=atlas,
        plan=plan,
        iv=iv,
        svd=svd,
        ablation=ablation,
        cards=cards,
        model_name=model_name,
        notes=notes,
        trace=trace,
    )
    if draft or agent is not None or slm or endpoint:
        from llmintent.anatomy.guide import draft_anatomy_report

        guide = draft_anatomy_report(
            report, agent=agent, slm=slm, endpoint=endpoint
        )
        report.draft = guide.markdown
        notes.extend(guide.notes)
        report.notes = notes
    return report
