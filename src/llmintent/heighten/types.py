"""Types for heightened / focused reasoning via forced retrace."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from llmintent.trajectory import TrajectoryMapping


class RetraceMode(str, Enum):
    """How to scaffold a self-retrace prompt."""

    EXPLICIT = "explicit_retrace"
    CONCEPT_ANCHOR = "concept_anchor"
    PIVOT_REPLAY = "pivot_replay"
    CORRECTION = "correction"
    FOCUSED_COT = "focused_cot"


@dataclass
class FocusMetrics:
    """
    Quantify how concentrated reasoning is across layers and modules.

    Higher focus_score → reasoning is layer-concentrated, concept-peaked,
    and not dominated by ideation/meta-monitoring.
    """

    reasoning_concentration: float
    concept_peakiness: float
    reasoning_ideation_ratio: float
    meta_load: float
    dispersion_index: float
    motor_prematurity: float
    focus_score: float
    needs_retrace: bool
    dominant_unfocused_layers: list[int] = field(default_factory=list)
    recommended_focus_layers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reasoning_concentration": self.reasoning_concentration,
            "concept_peakiness": self.concept_peakiness,
            "reasoning_ideation_ratio": self.reasoning_ideation_ratio,
            "meta_load": self.meta_load,
            "dispersion_index": self.dispersion_index,
            "motor_prematurity": self.motor_prematurity,
            "focus_score": self.focus_score,
            "needs_retrace": self.needs_retrace,
            "dominant_unfocused_layers": self.dominant_unfocused_layers,
            "recommended_focus_layers": self.recommended_focus_layers,
        }


@dataclass
class RetracePlan:
    """Generated retrace scaffold and target layers."""

    mode: RetraceMode
    baseline_prompt: str
    anchor_prompt: str
    retrace_prompt: str
    focused_prompt: str | None
    concepts: list[str]
    retrace_layers: list[int]
    rationale: str

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "baseline_prompt": self.baseline_prompt,
            "anchor_prompt": self.anchor_prompt,
            "retrace_prompt": self.retrace_prompt,
            "focused_prompt": self.focused_prompt,
            "concepts": self.concepts,
            "retrace_layers": self.retrace_layers,
            "rationale": self.rationale,
        }


@dataclass
class HeightenedReasoningResult:
    """Before/after comparison of focus metrics across retrace."""

    prompt: str
    anchor_prompt: str
    plan: RetracePlan
    baseline_focus: FocusMetrics
    retrace_focus: FocusMetrics
    focused_focus: FocusMetrics | None
    baseline_mapping: TrajectoryMapping
    retrace_mapping: TrajectoryMapping
    focus_gain: dict[str, float]
    intervention_layers: list[int] = field(default_factory=list)
    steering_report: dict[str, float] = field(default_factory=dict)

    @property
    def heightening_successful(self) -> bool:
        return self.focus_gain.get("focus_score_delta", 0.0) > 0.05

    def summary(self) -> pd.DataFrame:
        rows = [
            {"phase": "baseline", **self.baseline_focus.to_dict()},
            {"phase": "retrace", **self.retrace_focus.to_dict()},
        ]
        if self.focused_focus:
            rows.append({"phase": "focused", **self.focused_focus.to_dict()})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "anchor_prompt": self.anchor_prompt,
            "plan": self.plan.to_dict(),
            "baseline_focus": self.baseline_focus.to_dict(),
            "retrace_focus": self.retrace_focus.to_dict(),
            "focused_focus": self.focused_focus.to_dict() if self.focused_focus else None,
            "focus_gain": self.focus_gain,
            "intervention_layers": self.intervention_layers,
            "steering_report": self.steering_report,
            "heightening_successful": self.heightening_successful,
        }
