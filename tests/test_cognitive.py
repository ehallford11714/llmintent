"""Tests for twin Barlow minimization and cognitive scoring."""

import torch

from llmintent.cognitive.identity import identity_layer_scores
from llmintent.cognitive.meta_reasoning import meta_reasoning_layer_scores
from llmintent.cognitive.reasoning import reasoning_layer_scores
from llmintent.kernels.barlow import barlow_cross_correlation, barlow_twins_loss, minimize_twin_barlow


def test_barlow_identity_target():
    z = torch.randn(8, 16)
    c = barlow_cross_correlation(z, z)
    loss = barlow_twins_loss(c)
    assert loss.item() < 1.0


def test_minimize_twin_barlow_runs():
    h_a = torch.randn(6, 32)
    h_b = h_a + 0.1 * torch.randn(6, 32)
    kl = torch.tensor([0.1, 0.2, 0.5, 1.0, 0.8, 0.3])
    w, metrics = minimize_twin_barlow(h_a, h_b, kl, proj_dim=8, steps=10)
    assert w.shape[0] == 32
    assert w.shape[1] <= 8
    assert "barlow_loss" in metrics


def test_cognitive_scores_shape():
    n = 5
    kl = torch.tensor([0.1, 0.4, 1.0, 0.6, 0.2])
    diag = torch.linspace(0.9, 0.5, n)
    off = torch.linspace(0.1, 0.9, n)
    occ = torch.tensor([1.0, 2.0, 5.0, 3.0, 1.5])
    ent = torch.tensor([2.0, 3.0, 4.0, 3.5, 2.5])
    motor = torch.linspace(0.1, 0.95, n)

    id_s = identity_layer_scores(kl, diag, motor_alignment=motor)
    re_s = reasoning_layer_scores(kl, occ, entropy=ent)
    meta_s = meta_reasoning_layer_scores(kl, off)

    assert id_s.shape == (n,)
    assert re_s.shape == (n,)
    assert meta_s.shape == (n,)
    assert float(re_s.max()) > 0
