"""Types and epistemic constants for vendored latent inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EPISTEMIC_CAVEATS: tuple[str, ...] = (
    "Inspects model-internal correlates / probes / heuristics — not human thoughts.",
    "Probe and SAE labels are hypothesized correlates, not proven goals or beliefs.",
    "Verbalized chain-of-thought may be unfaithful to internal computation.",
    "Layer saliency in MVP is heuristic unless causal patching evidence is provided.",
    "Do not claim to 'read the mind' of the model beyond the stated evidence class.",
)

INTENT_TAGS: tuple[str, ...] = (
    "request_info",
    "request_action",
    "express_goal",
    "express_constraint",
    "refuse_or_hedge",
    "plan_or_reason",
    "social_affect",
    "other",
)


@dataclass
class IntentHypothesis:
    tag: str
    score: float
    method: str
    layer: int | None = None
    evidence: str = ""
    confidence: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayerSaliency:
    layer: int
    score: float
    source: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
