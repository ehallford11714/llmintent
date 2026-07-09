"""Smoke tests for Live visualization panel (no network)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from llmintent.live.viz_panel import build_live_mapping
from llmintent.trajectory import TrajectoryMapping
from llmintent.viz.backend import new_figure
from llmintent.viz.maps import plot_trajectory_map


def _sample_mapping() -> TrajectoryMapping:
    n = 6
    layers = pd.DataFrame(
        {
            "layer": range(n),
            "entropy": [4.0, 3.0, 2.0, 1.5, 1.0, 0.5],
            "kl_divergence": [0.1, 0.3, 0.6, 0.9, 0.7, 0.4],
            "intensity": [0.2, 0.4, 0.6, 0.8, 0.7, 0.5],
            "reasoning": [0.1, 0.3, 0.5, 0.7, 0.6, 0.4],
            "meta_reasoning": [0.05, 0.15, 0.3, 0.5, 0.45, 0.3],
        }
    )
    return TrajectoryMapping(
        prompt="test",
        twin_b=None,
        model_name="mock",
        num_layers=n,
        layers=layers,
        pivots={"inference_pivot": 3},
    )


def test_viz_panel_plot_trajectory_from_mapping():
    mapping = _sample_mapping()
    fig, ax = new_figure()
    plot_trajectory_map(mapping, ax=ax)
    assert ax.figure is not None
    plt.close(fig)


def test_build_live_mapping_delegates(monkeypatch):
    """build_live_mapping forwards to build_trajectory_mapping with bundle from pipeline."""
    captured: dict = {}

    def fake_build(bundle, prompt, **kwargs):
        captured["bundle"] = bundle
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return _sample_mapping()

    monkeypatch.setattr("llmintent.live.viz_panel.build_trajectory_mapping", fake_build)

    class FakeBundle:
        pass

    class FakeSession:
        bundle = FakeBundle()

    class FakePipe:
        session = FakeSession()

    build_live_mapping(
        FakePipe(),  # type: ignore[arg-type]
        "hello world",
        twin_b="twin",
        concepts=["hello"],
        include_cognitive=True,
        include_concepts=True,
    )
    assert captured["prompt"] == "hello world"
    assert captured["kwargs"]["twin_b"] == "twin"
    assert captured["kwargs"]["concepts"] == ["hello"]
    assert captured["kwargs"]["include_cognitive"] is True
