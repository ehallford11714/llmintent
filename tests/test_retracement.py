"""Tests for Retracement Transformer (unit + optional slow perplexity)."""

import math

import torch

from llmintent.retracement.blocks import FocusGate, RetraceMerge, compute_self_focus_vector
from llmintent.retracement.config import RetracementConfig, RetracementMode
from llmintent.retracement.perplexity import load_eval_texts


def test_focus_gate_changes_hidden():
    gate = FocusGate(64, coefficient=0.5)
    vec = torch.randn(64)
    gate.set_focus_vector(vec)
    h = torch.randn(2, 10, 64)
    out = gate(h)
    assert out.shape == h.shape
    assert not torch.allclose(out, h)


def test_retrace_merge_blend():
    merge = RetraceMerge(blend=0.5)
    snap = torch.ones(1, 4, 32)
    merge.set_snapshot(snap)
    h = torch.zeros(1, 4, 32)
    out = merge(h)
    assert torch.allclose(out, torch.full_like(h, 0.5))


def test_self_focus_vector_unit_norm():
    h = torch.randn(1, 8, 128)
    v = compute_self_focus_vector(h)
    assert v.shape == (128,)
    assert math.isclose(float(torch.norm(v)), 1.0, rel_tol=1e-4)


def test_config_partition():
    cfg = RetracementConfig()
    s, w, n = cfg.partition(12)
    assert 0 < s < w < n


def test_resolve_layers_baseline_empty():
    cfg = RetracementConfig(mode=RetracementMode.BASELINE)
    assert cfg.resolve_layers(12) == []


def test_resolve_layers_extreme_covers_workspace():
    cfg = RetracementConfig(mode=RetracementMode.EXTREME)
    layers = cfg.resolve_layers(12)
    assert len(layers) >= 2


def test_fallback_corpus():
    texts = load_eval_texts(limit=5)
    assert len(texts) == 5
