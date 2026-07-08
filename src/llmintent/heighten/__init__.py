"""Heightened reasoning: force retrace to sharpen focused reasoning."""

from llmintent.heighten.cot_delta import compute_cot_delta
from llmintent.heighten.focus import compare_focus, compute_focus_metrics
from llmintent.heighten.framework import HeightenedReasoningFramework, heighten_reasoning
from llmintent.heighten.extreme import (
    ExtremeRetraceChain,
    ExtremeRetraceMode,
    build_extreme_retrace_chain,
    heighten_until_focused,
)
from llmintent.heighten.intervention import (
    apply_focus_steering,
    extract_reasoning_focus_vector,
    forward_with_focus_steering,
    steering_hooks,
)
from llmintent.heighten.retrace import build_focused_prompt, build_retrace_prompt, plan_retrace
from llmintent.heighten.types import (
    FocusMetrics,
    HeightenedReasoningResult,
    RetraceMode,
    RetracePlan,
)

__all__ = [
    "FocusMetrics",
    "HeightenedReasoningFramework",
    "HeightenedReasoningResult",
    "RetraceMode",
    "RetracePlan",
    "ExtremeRetraceChain",
    "ExtremeRetraceMode",
    "apply_focus_steering",
    "build_extreme_retrace_chain",
    "build_focused_prompt",
    "build_retrace_prompt",
    "compare_focus",
    "compute_cot_delta",
    "compute_focus_metrics",
    "extract_reasoning_focus_vector",
    "forward_with_focus_steering",
    "heighten_reasoning",
    "heighten_until_focused",
    "plan_retrace",
    "steering_hooks",
]
