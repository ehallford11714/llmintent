import pytest
import torch

from llmintent.metrics import calculate_sso_score, kl_divergence, shannon_entropy


def test_sso_score():
    assert calculate_sso_score(0.8, 0.2) == pytest.approx(0.6)
    assert calculate_sso_score(0.0, 0.0) == 0.0


def test_entropy():
    uniform = torch.ones(4) / 4
    assert shannon_entropy(uniform) == pytest.approx(1.386294, rel=1e-4)


def test_kl_divergence_identical():
    p = torch.tensor([0.5, 0.3, 0.2])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)
