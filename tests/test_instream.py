"""In-stream bind writes at entity sites and later tokens, not last-only."""

from __future__ import annotations

import torch
import torch.nn as nn

from llmintent.instream import _apply_to_hidden, bind_positions


def test_bind_positions_include_members_and_tail():
    spans = [
        {"position": 0, "token": "8"},
        {"position": 3, "token": "2"},
    ]
    pos = bind_positions(8, spans)
    assert pos == list(range(0, 8))


def test_apply_writes_slice_not_only_last():
    h = torch.zeros(1, 5, 4)
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = _apply_to_hidden(h, v, [0, 3, 4], gain=2.0)
    assert float(out[0, 0, 0]) == 2.0
    assert float(out[0, 3, 0]) == 2.0
    assert float(out[0, 4, 0]) == 2.0
    assert float(out[0, 1, 0]) == 0.0
    assert float(out[0, 2, 0]) == 0.0


def test_dummy_block_hook_touches_bound_positions():
    layer = nn.Identity()
    hidden = torch.zeros(1, 4, 3)
    vec = torch.ones(3)
    positions = [1, 2]

    def hook(_m, _i, out):
        return _apply_to_hidden(out, vec, positions, 1.0)

    h = layer.register_forward_hook(hook)
    try:
        y = layer(hidden)
    finally:
        h.remove()
    assert float(y[0, 1].sum()) == 3.0
    assert float(y[0, 0].sum()) == 0.0
