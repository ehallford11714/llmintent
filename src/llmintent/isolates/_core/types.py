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
