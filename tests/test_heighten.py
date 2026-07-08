"""Tests for Heightened Reasoning Framework (synthetic, no model load)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llmintent.heighten.focus import compare_focus, compute_focus_metrics
from llmintent.heighten.retrace import build_retrace_prompt, plan_retrace
from llmintent.heighten.types import RetraceMode
from llmintent.trajectory import TrajectoryMapping


def _focused_mapping() -> TrajectoryMapping:
    n = 8
    reasoning = np.array([0.1, 0.2, 0.3, 0.9, 0.85, 0.4, 0.2, 0.1])
    ideation = np.array([0.5, 0.4, 0.3, 0.1, 0.05, 0.1, 0.2, 0.3])
    layers = pd.DataFrame(
        {
            "layer": range(n),
            "reasoning": reasoning,
            "ideation": ideation,
            "meta_reasoning": [0.1] * n,
            "motor_alignment": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.95],
            "dominant_module": ["ideation", "ideation", "reasoning", "reasoning", "reasoning", "reasoning", "identity", "identity"],
            "concept_math_activation": [0.1, 0.2, 0.4, 0.95, 0.8, 0.3, 0.1, 0.05],
        }
    )
    return TrajectoryMapping(
        prompt="test",
        twin_b="twin",
        model_name="gpt2",
        num_layers=n,
        layers=layers,
        pivots={"inference_pivot": 3, "workspace_peak": 4},
    )


def _diffuse_mapping() -> TrajectoryMapping:
    n = 8
    layers = pd.DataFrame(
        {
            "layer": range(n),
            "reasoning": [0.3, 0.35, 0.32, 0.33, 0.31, 0.34, 0.3, 0.32],
            "ideation": [0.6, 0.55, 0.58, 0.57, 0.56, 0.59, 0.6, 0.58],
            "meta_reasoning": [0.5, 0.48, 0.52, 0.49, 0.51, 0.5, 0.47, 0.53],
            "motor_alignment": [0.7, 0.72, 0.71, 0.73, 0.74, 0.75, 0.9, 0.95],
            "dominant_module": ["ideation"] * 4 + ["meta_reasoning"] * 4,
            "concept_math_activation": [0.4, 0.42, 0.41, 0.39, 0.4, 0.38, 0.41, 0.4],
        }
    )
    return TrajectoryMapping(
        prompt="test",
        twin_b="twin",
        model_name="gpt2",
        num_layers=n,
        layers=layers,
        pivots={"inference_pivot": 2},
    )


def test_focused_mapping_has_higher_focus_score():
    focused = compute_focus_metrics(_focused_mapping())
    diffuse = compute_focus_metrics(_diffuse_mapping())
    assert focused.focus_score > diffuse.focus_score
    assert diffuse.needs_retrace is True


def test_compare_focus_delta():
    before = compute_focus_metrics(_diffuse_mapping())
    after = compute_focus_metrics(_focused_mapping())
    delta = compare_focus(before, after)
    assert delta["focus_score_delta"] > 0


def test_retrace_prompt_contains_concepts():
    prompt = build_retrace_prompt(
        "Question: 8 - 2 = ? Answer:",
        mode=RetraceMode.EXPLICIT,
        concepts=["subtraction", "eight"],
    )
    assert "retrace" in prompt.lower()
    assert "subtraction" in prompt


def test_plan_retrace_includes_pivot_layers():
    mapping = _diffuse_mapping()
    plan = plan_retrace(
        "Question: 8 - 2 = ?",
        "Question: 8 - 2 = ? Let's think step by step.",
        mapping,
        concepts=["subtraction"],
    )
    assert plan.retrace_prompt
    assert 2 in plan.retrace_layers or 3 in plan.retrace_layers
    assert "subtraction" in plan.concepts
