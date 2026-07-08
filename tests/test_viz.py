"""Tests for visualization suite (synthetic data, Agg backend)."""

from __future__ import annotations

import pandas as pd
import pytest

from llmintent.trajectory import TrajectoryMapping
from llmintent.viz import correlation, maps


def _sample_mapping() -> TrajectoryMapping:
    n = 8
    layers = pd.DataFrame(
        {
            "layer": range(n),
            "entropy": [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
            "kl_divergence": [0.1, 0.2, 0.4, 0.8, 1.2, 1.0, 0.6, 0.3],
            "intensity": [0.2, 0.3, 0.5, 0.7, 0.9, 0.85, 0.6, 0.4],
            "occupancy": [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
            "motor_alignment": [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99],
            "reasoning": [0.1, 0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3],
            "meta_reasoning": [0.05, 0.1, 0.2, 0.4, 0.6, 0.55, 0.4, 0.2],
            "ideation": [0.02, 0.05, 0.1, 0.2, 0.3, 0.25, 0.15, 0.1],
            "top_intent": ["a"] * n,
            "concept_subtraction_activation": [0.1, 0.2, 0.5, 0.9, 0.7, 0.4, 0.2, 0.1],
            "concept_eight_activation": [0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        }
    )
    return TrajectoryMapping(
        prompt="test",
        twin_b="twin",
        model_name="gpt2",
        num_layers=n,
        layers=layers,
        pivots={"inference_pivot": 4, "workspace_peak": 5},
    )


def _sample_block_semantics() -> dict:
    return {
        0: {"mlp": ["subtract", "number"], "attn": ["eight"]},
        1: {"mlp": ["subtract", "minus"], "attn": ["two"]},
        2: {"mlp": ["equals", "answer"], "attn": ["subtract"]},
    }


@pytest.fixture
def mapping():
    return _sample_mapping()


@pytest.fixture
def block_semantics():
    return _sample_block_semantics()


def test_morpheme_map_saves(tmp_path, block_semantics):
    path = tmp_path / "morpheme.png"
    maps.save_morpheme_map(block_semantics, str(path))
    assert path.exists()
    assert path.stat().st_size > 500


def test_trajectory_map_saves(tmp_path, mapping):
    path = tmp_path / "trajectory.png"
    maps.save_trajectory_map(mapping, str(path))
    assert path.exists()


def test_concept_correlation_matrix(mapping):
    corr = correlation.build_concept_correlation_matrix(mapping)
    assert corr.shape[0] >= 2
    assert corr.loc["concept_subtraction_activation", "concept_subtraction_activation"] == pytest.approx(1.0)


def test_reasoning_trace_correlation(mapping):
    corr = correlation.build_reasoning_trace_correlation(mapping)
    assert "entropy" in corr.index
    assert "reasoning" in corr.columns


def test_correlation_plot_saves(tmp_path, mapping):
    path = tmp_path / "corr.png"
    correlation.save_concept_correlation(mapping, str(path))
    assert path.exists()


def test_trajectory_animation_saves(tmp_path, mapping):
    pytest.importorskip("PIL")
    path = tmp_path / "anim.gif"
    from llmintent.viz.animate import animate_trajectory_maturation

    animate_trajectory_maturation(mapping, save_path=str(path), interval_ms=50)
    assert path.exists()
