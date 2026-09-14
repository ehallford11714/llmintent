"""Anatomy report: what each LLM region handles, how it is wired, ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmintent.anatomy.ablation import AblationResult, ablate_model, plant_and_ablate
from llmintent.anatomy.atlas import REGIONS, Atlas
from llmintent.anatomy.compile import RegionPlan, compile_regions
from llmintent.anatomy.connectome import default_atlas, literature_region_connectome
from llmintent.anatomy.iv_engine import AnatomyIVResult, iv_from_text
from llmintent.anatomy.svd_map import SVDAnatomy, map_activations, map_weights


@dataclass
class RegionCard:
    id: str
    handles: str
    fly_neuropil: str
    band: str
    integrates_with: list[str]
    occupancy: float
    compiled: bool
    depth: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "handles": self.handles,
            "fly_neuropil": self.fly_neuropil,
            "band": self.band,
            "integrates_with": list(self.integrates_with),
            "occupancy": round(self.occupancy, 4),
            "compiled": self.compiled,
            "depth": list(self.depth),
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
            "| Region | Handles | Band | Integrates with | Occupancy | Compiled |",
            "|--------|---------|------|-----------------|-----------|----------|",
        ]
        for c in self.cards:
            integ = ", ".join(c.integrates_with) or "—"
            flag = "yes" if c.compiled else ""
            lines.append(
                f"| `{c.id}` | {c.handles} | {c.band} | {integ} | "
                f"{c.occupancy:.3f} | {flag} |"
            )
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


def _cards(atlas: Atlas, occupancy: dict[str, float], compiled: set[str]) -> list[RegionCard]:
    out: list[RegionCard] = []
    for r in REGIONS:
        out.append(
            RegionCard(
                id=r.id,
                handles=r.handles,
                fly_neuropil=r.fly_neuropil,
                band=r.band,
                integrates_with=list(atlas.integrates.get(r.id, ())),
                occupancy=float(occupancy.get(r.id, 0.0)),
                compiled=r.id in compiled,
                depth=r.depth,
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
) -> AnatomyReport:
    """
    Map LLM anatomy for ``text``.

    Offline (no bundle): compile + connectome IV + planted A vs B ablation.
    With a model bundle: add activation SVD occupancy and residual-stream ablation.
    """
    atlas = default_atlas()
    plan, iv = iv_from_text(text, mock_iv=mock_iv, seed=seed)
    occupancy = dict(plan.occupancy())
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

    cards = _cards(atlas, occupancy, set(plan.regions))
    notes.append(
        f"Connectome source: {literature_region_connectome().source}."
    )
    return AnatomyReport(
        text=text,
        atlas=atlas,
        plan=plan,
        iv=iv,
        svd=svd,
        ablation=ablation,
        cards=cards,
        model_name=model_name,
        notes=notes,
    )
