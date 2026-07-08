"""Heightened Reasoning Framework — force retrace to sharpen focused reasoning."""

from __future__ import annotations

from llmintent.cognitive.orchestrator import build_cognitive_module_profile
from llmintent.heighten.cot_delta import compute_cot_delta
from llmintent.heighten.focus import compare_focus, compute_focus_metrics
from llmintent.heighten.intervention import apply_focus_steering
from llmintent.heighten.retrace import plan_retrace
from llmintent.heighten.types import (
    FocusMetrics,
    HeightenedReasoningResult,
    RetraceMode,
    RetracePlan,
)
from llmintent.jspace.transport import TransportMaps
from llmintent.models import ModelBundle
from llmintent.trajectory import TrajectoryMapping, build_trajectory_mapping


class HeightenedReasoningFramework:
    """
    Heighten reasoning by forcing a self-retrace and measuring focus gain.

    Pipeline:
    1. Diagnose focus on baseline vs anchor (CoT) trajectory
    2. If diffuse → generate retrace scaffold prompt
    3. Re-analyze with retrace twin → measure focus gain
    4. Optionally apply activation steering at reasoning layers

    Focused reasoning = high reasoning concentration, peaked concept activation,
    low ideation/meta dispersion, late motor commit.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        transport: TransportMaps | None = None,
        focus_threshold: float = 0.45,
    ) -> None:
        self.bundle = bundle
        self.transport = transport
        self.focus_threshold = focus_threshold

    def diagnose_focus(
        self,
        prompt: str,
        *,
        anchor_prompt: str | None = None,
        concepts: list[str] | None = None,
    ) -> tuple[FocusMetrics, TrajectoryMapping]:
        anchor = anchor_prompt or prompt
        mapping = build_trajectory_mapping(
            self.bundle,
            prompt,
            twin_b=anchor if anchor != prompt else None,
            transport=self.transport,
            concepts=concepts,
            include_cognitive=anchor != prompt,
        )
        return compute_focus_metrics(mapping, focus_threshold=self.focus_threshold), mapping

    def build_plan(
        self,
        prompt: str,
        anchor_prompt: str,
        *,
        mode: RetraceMode = RetraceMode.EXPLICIT,
        concepts: list[str] | None = None,
    ) -> RetracePlan:
        _, mapping = self.diagnose_focus(prompt, anchor_prompt=anchor_prompt, concepts=concepts)
        return plan_retrace(prompt, anchor_prompt, mapping, mode=mode, concepts=concepts)

    def heighten(
        self,
        prompt: str,
        *,
        anchor_prompt: str | None = None,
        concepts: list[str] | None = None,
        mode: RetraceMode = RetraceMode.EXPLICIT,
        apply_steering: bool = False,
        steering_coefficient: float = 0.5,
    ) -> HeightenedReasoningResult:
        anchor = anchor_prompt or prompt
        cot_delta = None
        cognitive = None
        if anchor != prompt:
            cot_delta = compute_cot_delta(self.bundle, prompt, anchor)
            cognitive = build_cognitive_module_profile(
                self.bundle,
                prompt,
                anchor,
                transport=self.transport,
                cot_delta=cot_delta,
            )

        baseline_mapping = build_trajectory_mapping(
            self.bundle,
            prompt,
            twin_b=anchor if anchor != prompt else None,
            transport=self.transport,
            concepts=concepts,
            include_cognitive=anchor != prompt,
        )
        baseline_focus = compute_focus_metrics(baseline_mapping, focus_threshold=self.focus_threshold)

        plan = plan_retrace(
            prompt,
            anchor,
            baseline_mapping,
            mode=mode,
            concepts=concepts,
        )

        retrace_mapping = build_trajectory_mapping(
            self.bundle,
            prompt,
            twin_b=plan.retrace_prompt,
            transport=self.transport,
            concepts=concepts or plan.concepts,
            include_cognitive=True,
        )
        retrace_focus = compute_focus_metrics(retrace_mapping, focus_threshold=self.focus_threshold)

        focused_focus = None
        if plan.focused_prompt:
            focused_mapping = build_trajectory_mapping(
                self.bundle,
                plan.focused_prompt,
                twin_b=plan.retrace_prompt,
                transport=self.transport,
                concepts=concepts or plan.concepts,
                include_cognitive=True,
            )
            focused_focus = compute_focus_metrics(focused_mapping, focus_threshold=self.focus_threshold)

        focus_gain = compare_focus(baseline_focus, retrace_focus)
        if focused_focus is not None:
            focus_gain["focused_vs_baseline"] = compare_focus(
                baseline_focus, focused_focus
            )["focus_score_delta"]

        steering_report: dict = {}
        intervention_layers = plan.retrace_layers

        if apply_steering and intervention_layers:
            steer = apply_focus_steering(
                self.bundle,
                prompt,
                anchor,
                plan.retrace_prompt,
                layer_indices=intervention_layers,
                concepts=concepts or plan.concepts,
                coefficient=steering_coefficient,
                transport=self.transport,
                cognitive=cognitive,
            )
            steering_report = steer.to_dict()
            focus_gain["steering_focus_delta"] = steer.focus_gain.get("focus_score_delta", 0.0)

        return HeightenedReasoningResult(
            prompt=prompt,
            anchor_prompt=anchor,
            plan=plan,
            baseline_focus=baseline_focus,
            retrace_focus=retrace_focus,
            focused_focus=focused_focus,
            baseline_mapping=baseline_mapping,
            retrace_mapping=retrace_mapping,
            focus_gain=focus_gain,
            intervention_layers=intervention_layers,
            steering_report=steering_report,
        )


def heighten_reasoning(
    bundle: ModelBundle,
    prompt: str,
    *,
    anchor_prompt: str | None = None,
    concepts: list[str] | None = None,
    mode: RetraceMode = RetraceMode.EXPLICIT,
    transport: TransportMaps | None = None,
    apply_steering: bool = False,
) -> HeightenedReasoningResult:
    """Functional API for the Heightened Reasoning Framework."""
    framework = HeightenedReasoningFramework(bundle, transport=transport)
    return framework.heighten(
        prompt,
        anchor_prompt=anchor_prompt,
        concepts=concepts,
        mode=mode,
        apply_steering=apply_steering,
    )
