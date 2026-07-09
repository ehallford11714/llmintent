"""Tests for twin Barlow — repeated optimization must not autograd-fail."""

import torch

from llmintent.kernels.barlow import minimize_twin_barlow


def test_minimize_twin_barlow_twice():
    """Simulates Live heighten calling Barlow twice in one session."""
    h = torch.randn(12, 64)
    kl = torch.rand(12)

    w1, m1 = minimize_twin_barlow(h, h + 0.1, kl, steps=5, proj_dim=8)
    w2, m2 = minimize_twin_barlow(h, h + 0.2, kl, steps=5, proj_dim=8)

    assert w1.shape == w2.shape == (64, 8)
    assert "barlow_loss" in m1 and "barlow_loss" in m2
