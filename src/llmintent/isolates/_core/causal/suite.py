"""LayerCausalSuite — identify → motifs → trajectory → indication/IV report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from llmintent.isolates._core.causal.features import MotifFeatureTable, build_feature_frame
from llmintent.isolates._core.causal.iv_layers import (
    CausalEdgeEstimate,
    IndicationScore,
    estimate_indication,
    estimate_layer_iv,
    layer_causation_matrix,
    layer_indication_matrix,
)
from llmintent.isolates._core.identify import identify_isolates
from llmintent.isolates._core.motifs import _layer_sort_key, form_motifs
from llmintent.isolates._core.trajectory import trajectory_from_motifs
from llmintent.isolates._core.types import (
    ABSTRACT_LAYERS,
    Isolate,
    IsolateReport,
    Motif,
    ReasoningTrajectory,
)

CAVEATS = [
    "Indication (association) is not causation. High layer->Y correlation does not imply the layer causes Y.",
    "IV / 2SLS requires: relevance (Z correlates with X), exclusion (Z affects Y only through X), and no Z-confounder of Y.",
    "Abstract L0-L4 layers are a reasoning scaffold unless bound to model residual indices.",
    "Bootstrap rows from a single text are synthetic - exploratory only, not population inference.",
    "Weak instruments (low first-stage F) bias beta_IV toward OLS; treat weak edges cautiously.",
]


@dataclass
class LayerCausalResult:
    """Full bridge result: isolates, motifs, trajectory, indication, causation."""

    isolates: list[Isolate] = field(default_factory=list)
    motifs: list[Motif] = field(default_factory=list)
    trajectory: ReasoningTrajectory | None = None
    feature_table: MotifFeatureTable | None = None
    indications: list[IndicationScore] = field(default_factory=list)
    causation_edges: list[CausalEdgeEstimate] = field(default_factory=list)
    indication_by_layer: dict[int, float] = field(default_factory=dict)
    causation_by_layer: dict[int, float] = field(default_factory=dict)
    outcome_hint: str | None = None
    backend: str = "rule"
    iv_method_notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    input_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "isolates": [i.to_dict() for i in self.isolates],
            "motifs": [m.to_dict() for m in self.motifs],
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "feature_table": self.feature_table.to_dict() if self.feature_table else None,
            "indications": [s.to_dict() for s in self.indications],
            "causation_edges": [e.to_dict() for e in self.causation_edges],
            "indication_by_layer": {str(k): v for k, v in self.indication_by_layer.items()},
            "causation_by_layer": {str(k): v for k, v in self.causation_by_layer.items()},
            "outcome_hint": self.outcome_hint,
            "backend": self.backend,
            "iv_method_notes": list(self.iv_method_notes),
            "caveats": list(self.caveats),
            "metadata": dict(self.metadata),
            "input_summary": self.input_summary,
            "n_isolates": len(self.isolates),
            "n_motifs": len(self.motifs),
            "n_indication": len(self.indications),
            "n_causation": len(self.causation_edges),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Layer Causal Report",
            "",
            f"**Backend:** `{self.backend}`  ",
            f"**Input:** {self.input_summary or '(none)'}  ",
            f"**Outcome hint:** {self.outcome_hint or '(auto)'}  ",
            f"**Isolates:** {len(self.isolates)} · **Motifs:** {len(self.motifs)} · "
            f"**Indication edges:** {len(self.indications)} · "
            f"**IV edges:** {len(self.causation_edges)}",
            "",
        ]

        lines.append("## Caveats")
        for c in self.caveats:
            lines.append(f"- {c}")
        lines.append("")

        lines.append("## Isolates by layer")
        by_layer: dict[int, list[Isolate]] = {}
        for iso in self.isolates:
            L = _layer_sort_key(iso.layer)
            by_layer.setdefault(L, []).append(iso)
        for L in sorted(by_layer):
            name = ABSTRACT_LAYERS.get(L, f"L{L}")
            lines.append(f"### {name}")
            for iso in by_layer[L]:
                typ = iso.typology.value if hasattr(iso.typology, "value") else str(iso.typology)
                lines.append(
                    f"- `{iso.id}` **{typ}** (conf={iso.confidence:.2f}): {iso.label}"
                )
            lines.append("")

        if self.motifs:
            lines.append("## Motifs")
            for m in self.motifs:
                typ = m.typology.value if hasattr(m.typology, "value") else str(m.typology)
                lines.append(
                    f"- `{m.id}` **{typ}** [{m.pattern}] layers={m.layers} "
                    f"support={m.support:.2f}"
                )
            lines.append("")

        if self.trajectory:
            lines.append("## Trajectory")
            lines.append(self.trajectory.summary_markdown or "")
            if self.trajectory.ascii_diagram:
                lines.append("")
                lines.append("```")
                lines.append(self.trajectory.ascii_diagram)
                lines.append("```")
            lines.append("")

        lines.append("## Indication matrix (layer → output association)")
        lines.append("")
        lines.append("| Layer | max |r| | Interpretation |")
        lines.append("|------|--------|----------------|")
        for L, v in sorted(self.indication_by_layer.items()):
            name = ABSTRACT_LAYERS.get(L, f"L{L}")
            lines.append(f"| {name} | {v:.3f} | indicates Y (assoc.) |")
        if not self.indication_by_layer:
            lines.append("| — | — | _no associations above threshold_ |")
        lines.append("")
        if self.indications:
            lines.append("Top feature→Y associations:")
            for s in self.indications[:8]:
                lines.append(
                    f"- `{s.source}` @L{s.layer} → `{s.target}`: r={s.association:+.3f}"
                )
            lines.append("")

        lines.append("## Causation (IV edges)")
        lines.append("")
        if self.iv_method_notes:
            for n in self.iv_method_notes:
                lines.append(f"- _{n}_")
            lines.append("")
        lines.append("| X (endogenous) | Y | Z (instrument) | beta_IV | F1 | method |")
        lines.append("|----------------|---|----------------|---------|----|--------|")
        for e in self.causation_edges[:12]:
            weak = " (weak)" if e.weak_instrument else ""
            lines.append(
                f"| `{e.source}`@L{e.layer_x} | `{e.target}` | "
                f"`{e.instrument}`@L{e.layer_z} | {e.beta_iv:+.4f}{weak} | "
                f"{e.first_stage_f:.1f} | {e.method} |"
            )
        if not self.causation_edges:
            lines.append("| — | — | — | — | — | _none_ |")
        lines.append("")
        lines.append("Layer causation summary (max |beta_IV| by endogenous layer):")
        for L, v in sorted(self.causation_by_layer.items()):
            name = ABSTRACT_LAYERS.get(L, f"L{L}")
            lines.append(f"- **{name}**: {v:.4f}")
        lines.append("")
        lines.append(
            "> Compare indication vs causation: a layer may **indicate** Y "
            "(high |r|) without **causing** Y (near-zero / weak beta_IV), or the reverse "
            "when confounding masks association but IV recovers an effect."
        )
        lines.append("")
        return "\n".join(lines)


class LayerCausalSuite:
    """
    End-to-end bridge: text/features → isolates → motifs → trajectory →
    feature table → indication + IV causation report.
    """

    def __init__(
        self,
        *,
        text: str | None = None,
        features: Any = None,
        graph: Any = None,
        isolates: Sequence[Isolate] | None = None,
        backend: str = "rule",
    ) -> None:
        self.text = text
        self.features = features
        self.graph = graph
        self._isolates = list(isolates) if isolates is not None else None
        self.backend = backend
        self._last: LayerCausalResult | None = None

    @classmethod
    def from_text(cls, text: str, *, backend: str = "rule") -> "LayerCausalSuite":
        return cls(text=text, backend=backend)

    @classmethod
    def from_isolates(cls, isolates: Sequence[Isolate], *, backend: str = "rule") -> "LayerCausalSuite":
        return cls(isolates=isolates, backend=backend)

    @classmethod
    def from_report(cls, report: IsolateReport) -> "LayerCausalSuite":
        return cls(isolates=report.isolates, backend=report.backend)

    def run(
        self,
        *,
        outcome_hint: str | None = None,
        outcome: Sequence[float] | Mapping[str, float] | float | None = None,
        n_bootstrap: int = 48,
        seed: int = 17,
        mock_iv: bool = False,
        min_indication: float = 0.05,
    ) -> LayerCausalResult:
        if self._isolates is None:
            isolates = identify_isolates(
                text=self.text,
                features=self.features,
                graph=self.graph,
                backend=self.backend,
            )
        else:
            isolates = list(self._isolates)

        motifs = form_motifs(isolates)
        traj = trajectory_from_motifs(motifs, isolates)
        table = build_feature_frame(
            isolates,
            motifs,
            outcome=outcome,
            outcome_hint=outcome_hint,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        indications = estimate_indication(table, min_abs=min_indication)
        edges, iv_notes = estimate_layer_iv(table, mock=mock_iv)

        summary = ""
        if self.text:
            summary = self.text if len(self.text) <= 120 else self.text[:117] + "..."
        elif self.features is not None:
            summary = "features(...)"
        elif self.graph is not None:
            summary = "graph(...)"
        else:
            summary = f"{len(isolates)} isolates"

        result = LayerCausalResult(
            isolates=isolates,
            motifs=motifs,
            trajectory=traj,
            feature_table=table,
            indications=indications,
            causation_edges=edges,
            indication_by_layer=layer_indication_matrix(indications),
            causation_by_layer=layer_causation_matrix(edges),
            outcome_hint=outcome_hint,
            backend=self.backend,
            iv_method_notes=list(table.notes) + list(iv_notes),
            caveats=list(CAVEATS) + list(traj.caveats),
            metadata={
                "n_bootstrap": n_bootstrap,
                "seed": seed,
                "mock_iv": mock_iv,
                "feature_columns": table.columns,
            },
            input_summary=summary,
        )
        self._last = result
        return result

    def report_markdown(self) -> str:
        if self._last is None:
            raise RuntimeError("Call run() before report_markdown()")
        return self._last.to_markdown()

    def report_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        if self._last is None:
            raise RuntimeError("Call run() before report_json()")
        payload = json.dumps(self._last.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload
