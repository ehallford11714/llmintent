"""Core types for isolates, motifs, and reasoning trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TypologyLabel(str, Enum):
    """Taxonomy of isolate meaning / role."""

    LEXICAL = "lexical"
    AFFECTIVE = "affective"
    INSTRUMENTAL = "instrumental"
    CONFOUNDER = "confounder"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    NOISE = "noise"
    LATENT_FEATURE = "latent_feature"
    ORPHAN_NODE = "orphan_node"
    ACTION = "action"
    OUTCOME = "outcome"
    UNKNOWN = "unknown"


class IsolateKind(str, Enum):
    """Source modality of an isolate."""

    TEXT = "text"
    FEATURE = "feature"
    GRAPH = "graph"
    LAYER = "layer"


class MotifTypology(str, Enum):
    """Structural / typed motif classes."""

    CO_OCCURRENCE = "co_occurrence"
    SEQUENCE = "sequence"
    CHAIN = "chain"
    TRIANGLE = "triangle"
    TYPED_PATH = "typed_path"
    LAYER_BRIDGE = "layer_bridge"
    UNKNOWN = "unknown"


class TrajectoryRole(str, Enum):
    """Role of a motif/step in a reasoning trajectory."""

    EARLY_LEXICAL = "early_lexical"
    MID_LATENT = "mid_latent"
    LATE_GOAL = "late_goal"
    BRIDGE = "bridge"
    NOISE = "noise"
    UNKNOWN = "unknown"


# Abstract reasoning layers (always available offline).
ABSTRACT_LAYERS: dict[int, str] = {
    0: "L0_surface_lexical",
    1: "L1_semantic_binding",
    2: "L2_latent_workspace",
    3: "L3_goal_constraint",
    4: "L4_action_outcome",
}

# Default typed motif templates (typology sequences).
TYPED_MOTIF_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("goal", "constraint", "action"),
    ("affective", "instrumental", "outcome"),
    ("lexical", "latent_feature", "goal"),
    ("constraint", "instrumental", "outcome"),
    ("goal", "instrumental", "outcome"),
    ("affective", "constraint", "action"),
)


@dataclass
class TextSpan:
    """Contiguous character span in a source document."""

    start: int
    end: int
    surface: str
    sentence_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "surface": self.surface,
            "sentence_index": self.sentence_index,
        }


@dataclass
class Isolate:
    """A separable unit of intent / meaning / activation."""

    id: str
    kind: IsolateKind | str
    label: str
    typology: TypologyLabel | str = TypologyLabel.UNKNOWN
    confidence: float = 0.0
    rationale: str = ""
    span: tuple[int, int] | None = None
    layer: int | str | None = None
    layer_name: str | None = None
    source: str = "rule"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = _enum_val(self.kind)
        d["typology"] = _enum_val(self.typology)
        if self.span is not None:
            d["span"] = list(self.span)
        return d


@dataclass
class SpanIsolate:
    """Isolate bound to a contiguous text span — a hoppable creative stepping-stone.

    Used by :class:`~intentisolates.span_burst.CreativeBurstHopper` to jump
    span→span along a trajectory while preserving structural anchors
    (goal / constraint / outcome).
    """

    id: str
    typology: TypologyLabel | str
    text_span: TextSpan
    layer: int | str | None = None
    layer_name: str | None = None
    hop_weight: float = 1.0
    burst_affinity: float = 0.5
    confidence: float = 0.0
    rationale: str = ""
    source: str = "rule"
    protect: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def surface(self) -> str:
        return self.text_span.surface

    @property
    def start(self) -> int:
        return self.text_span.start

    @property
    def end(self) -> int:
        return self.text_span.end

    def to_isolate(self) -> Isolate:
        """Project to a plain :class:`Isolate` for motif / trajectory pipelines."""
        return Isolate(
            id=self.id,
            kind=IsolateKind.TEXT,
            label=self.surface,
            typology=self.typology,
            confidence=self.confidence,
            rationale=self.rationale,
            span=(self.start, self.end),
            layer=self.layer,
            layer_name=self.layer_name,
            source=self.source,
            metadata={
                **self.metadata,
                "hop_weight": self.hop_weight,
                "burst_affinity": self.burst_affinity,
                "protect": self.protect,
                "sentence_index": self.text_span.sentence_index,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "typology": _enum_val(self.typology),
            "text_span": self.text_span.to_dict(),
            "layer": _layer_val(self.layer) if self.layer is not None else None,
            "layer_name": self.layer_name,
            "hop_weight": self.hop_weight,
            "burst_affinity": self.burst_affinity,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source": self.source,
            "protect": self.protect,
            "metadata": dict(self.metadata),
        }


@dataclass
class BurstHop:
    """One hop in a creative-burst path."""

    from_id: str | None
    to_id: str
    mode: str
    score: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BurstPath:
    """Ordered hop trajectory for creative exploration."""

    seed_id: str
    hops: list[BurstHop] = field(default_factory=list)
    span_ids: list[str] = field(default_factory=list)
    typology_path: list[str] = field(default_factory=list)
    mode: str = "creative_burst"
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "hops": [h.to_dict() for h in self.hops],
            "span_ids": list(self.span_ids),
            "typology_path": list(self.typology_path),
            "mode": self.mode,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@dataclass
class CreativityReport:
    """CreativityMeter output: Guilford-inspired dimensions + reasoning fidelity."""

    diversity: float = 0.0
    novelty: float = 0.0
    flexibility: float = 0.0
    elaboration: float = 0.0
    constraint_fidelity: float = 0.0
    fluency: float = 0.0
    layer_monotonicity: float = 0.0
    creativity_score: float = 0.0
    reasoning_trace_score: float = 0.0
    tradeoff_product: float = 0.0
    tradeoff_harmonic: float = 0.0
    n_spans_scored: int = 0
    n_unique_typologies: int = 0
    typology_entropy: float = 0.0
    anchor_visit_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def tradeoff_points(self) -> list[dict[str, float]]:
        """Single (C, R) point for optional tradeoff plots."""
        return [
            {
                "creativity_score": self.creativity_score,
                "reasoning_trace_score": self.reasoning_trace_score,
                "product": self.tradeoff_product,
                "harmonic": self.tradeoff_harmonic,
            }
        ]


@dataclass
class Motif:
    """Recurring pattern / composition of isolates within or across layers."""

    id: str
    typology: MotifTypology | str
    member_ids: list[str]
    layers: list[int | str] = field(default_factory=list)
    support: float = 0.0
    confidence: float = 0.0
    trajectory_role: TrajectoryRole | str = TrajectoryRole.UNKNOWN
    pattern: str = ""
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["typology"] = _enum_val(self.typology)
        d["trajectory_role"] = _enum_val(self.trajectory_role)
        d["layers"] = [_layer_val(x) for x in self.layers]
        return d


@dataclass
class TrajectoryStep:
    """One ordered step in a reasoning trajectory."""

    index: int
    layer: int | str | None
    layer_name: str | None
    isolate_ids: list[str] = field(default_factory=list)
    motif_ids: list[str] = field(default_factory=list)
    role: TrajectoryRole | str = TrajectoryRole.UNKNOWN
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layer"] = _layer_val(self.layer) if self.layer is not None else None
        d["role"] = _enum_val(self.role)
        return d


@dataclass
class ReasoningTrajectory:
    """Ordered path of isolates/motifs across layers."""

    steps: list[TrajectoryStep] = field(default_factory=list)
    layer_path: list[int | str] = field(default_factory=list)
    motif_path: list[str] = field(default_factory=list)
    summary_markdown: str = ""
    mermaid: str = ""
    ascii_diagram: str = ""
    caveats: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "layer_path": [_layer_val(x) for x in self.layer_path],
            "motif_path": list(self.motif_path),
            "summary_markdown": self.summary_markdown,
            "mermaid": self.mermaid,
            "ascii_diagram": self.ascii_diagram,
            "caveats": list(self.caveats),
            "metadata": dict(self.metadata),
        }


@dataclass
class IsolateReport:
    """Aggregate report for identify / typology / motifs / trajectory."""

    isolates: list[Isolate] = field(default_factory=list)
    motifs: list[Motif] = field(default_factory=list)
    trajectory: ReasoningTrajectory | None = None
    backend: str = "rule"
    input_summary: str = ""
    caveats: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "isolates": [i.to_dict() for i in self.isolates],
            "motifs": [m.to_dict() for m in self.motifs],
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "backend": self.backend,
            "input_summary": self.input_summary,
            "caveats": list(self.caveats),
            "metadata": dict(self.metadata),
            "n_isolates": len(self.isolates),
            "n_motifs": len(self.motifs),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Isolate Report",
            "",
            f"**Backend:** `{self.backend}`  ",
            f"**Input:** {self.input_summary or '(none)'}  ",
            f"**Isolates:** {len(self.isolates)} · **Motifs:** {len(self.motifs)}",
            "",
        ]
        if self.caveats:
            lines.append("## Caveats")
            for c in self.caveats:
                lines.append(f"- {c}")
            lines.append("")

        lines.append("## Isolates")
        if not self.isolates:
            lines.append("_None detected._")
        else:
            for iso in self.isolates:
                layer = iso.layer_name or iso.layer
                lines.append(
                    f"- `{iso.id}` **{_enum_val(iso.typology)}** "
                    f"(kind={_enum_val(iso.kind)}, layer={layer}, "
                    f"conf={iso.confidence:.2f}): {iso.label}"
                )
                if iso.rationale:
                    lines.append(f"  - _{iso.rationale}_")
        lines.append("")

        if self.motifs:
            lines.append("## Motifs")
            for m in self.motifs:
                lines.append(
                    f"- `{m.id}` **{_enum_val(m.typology)}** "
                    f"[{m.pattern or ', '.join(m.member_ids)}] "
                    f"layers={m.layers} support={m.support:.2f} "
                    f"role={_enum_val(m.trajectory_role)}"
                )
                if m.rationale:
                    lines.append(f"  - _{m.rationale}_")
            lines.append("")

        if self.trajectory:
            lines.append("## Reasoning Trajectory")
            lines.append(self.trajectory.summary_markdown or "")
            if self.trajectory.ascii_diagram:
                lines.append("")
                lines.append("```")
                lines.append(self.trajectory.ascii_diagram)
                lines.append("```")
            if self.trajectory.mermaid:
                lines.append("")
                lines.append("```mermaid")
                lines.append(self.trajectory.mermaid)
                lines.append("```")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def _enum_val(v: Any) -> str:
    return v.value if isinstance(v, Enum) else str(v)


def _layer_val(v: Any) -> int | str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return v
    return str(v)
