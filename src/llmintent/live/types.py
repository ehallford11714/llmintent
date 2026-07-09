"""Shared types for the LLMIntent Live real-time suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LiveModelSpec:
    """Registered model for live inference."""

    key: str
    hf_name: str
    params_m: float
    description: str
    chat_template: bool = False
    default_retracement_mode: str = "focus_gate"
    recommended_for: str = "realtime"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "hf_name": self.hf_name,
            "params_m": self.params_m,
            "description": self.description,
            "chat_template": self.chat_template,
            "default_retracement_mode": self.default_retracement_mode,
            "recommended_for": self.recommended_for,
        }


@dataclass
class LiveAnalyzeResult:
    """Fast real-time analysis snapshot."""

    model_key: str
    prompt: str
    activation_layers: dict[str, int]
    focus_score: float | None
    needs_retrace: bool | None
    pivots: dict[str, int] = field(default_factory=dict)
    recommended_focus_layers: list[int] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "prompt": self.prompt,
            "activation_layers": self.activation_layers,
            "focus_score": self.focus_score,
            "needs_retrace": self.needs_retrace,
            "pivots": self.pivots,
            "recommended_focus_layers": self.recommended_focus_layers,
            "latency_ms": self.latency_ms,
        }


@dataclass
class LiveHeightenResult:
    """Real-time heighten / retrace outcome."""

    model_key: str
    prompt: str
    retrace_prompt: str
    focus_before: float | None
    focus_after: float | None
    focus_gain: float | None
    steering_applied: bool
    top_logits_shift: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "prompt": self.prompt,
            "retrace_prompt": self.retrace_prompt,
            "focus_before": self.focus_before,
            "focus_after": self.focus_after,
            "focus_gain": self.focus_gain,
            "steering_applied": self.steering_applied,
            "top_logits_shift": self.top_logits_shift,
            "latency_ms": self.latency_ms,
        }


@dataclass
class LiveGenerateResult:
    """Generation with optional retracement / steering."""

    model_key: str
    prompt: str
    completion: str
    retracement_mode: str
    steered: bool
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "prompt": self.prompt,
            "completion": self.completion,
            "retracement_mode": self.retracement_mode,
            "steered": self.steered,
            "latency_ms": self.latency_ms,
        }
