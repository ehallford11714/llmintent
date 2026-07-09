"""Tests for real-time layer activation streaming."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch

from llmintent.live.activation_stream import layer_metrics_from_states, LayerActivationSnapshot
from llmintent.live.viz_panel import (
    plot_activation_timeline,
    plot_live_activation_heatmap,
    plot_live_entropy_profile,
)


def _fake_states(n_layers: int = 6, hidden: int = 8) -> list[torch.Tensor]:
    return [torch.randn(1, 3, hidden) + i * 0.1 for i in range(n_layers)]


class _FakeHead(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.randn(32)


def test_layer_metrics_from_states(monkeypatch):
    class FakeBundle:
        device = "cpu"
        hidden_size = 8

    bundle = FakeBundle()
    states = _fake_states()

    monkeypatch.setattr(
        "llmintent.live.activation_stream.get_lm_head",
        lambda _b: _FakeHead(),
    )
    monkeypatch.setattr(
        "llmintent.live.activation_stream.build_numerical_pole",
        lambda _b: torch.ones(8),
    )
    monkeypatch.setattr(
        "llmintent.live.activation_stream.normalize_hidden",
        lambda _b, h: h,
    )

    df = layer_metrics_from_states(bundle, states)
    assert list(df.columns) == ["layer", "entropy", "intensity"]
    assert len(df) == len(states)


def test_plot_live_activation_helpers():
    metrics = pd.DataFrame(
        {
            "layer": range(4),
            "entropy": [4.0, 3.0, 2.0, 1.0],
            "intensity": [0.1, 0.3, 0.6, 0.9],
        }
    )
    fig, ax = plt.subplots()
    plot_live_activation_heatmap(metrics, step=1, token=" the", ax=ax)
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_live_entropy_profile(metrics, step=1, token=" the", ax=ax)
    plt.close(fig)


def test_plot_activation_timeline():
    snaps = [
        LayerActivationSnapshot(
            step=i,
            token="" if i == 0 else "a",
            token_id=1,
            layer_metrics=pd.DataFrame(
                {
                    "layer": [0, 1, 2],
                    "entropy": [4.0 - i * 0.2, 3.0 - i * 0.1, 2.0],
                    "intensity": [0.1 + i * 0.1, 0.2, 0.3],
                }
            ),
            cumulative_text="a" * i,
        )
        for i in range(3)
    ]
    fig, ax = plt.subplots()
    plot_activation_timeline(snaps, metric="entropy", ax=ax)
    plt.close(fig)
