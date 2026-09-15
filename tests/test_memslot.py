"""Memory slot sits before the last token and overwrites only that index."""

from __future__ import annotations

import torch
import torch.nn as nn

from llmintent.memslot import insert_memory_ids


def test_insert_memory_before_last():
    ids = torch.tensor([[10, 11, 12]])
    out, pos = insert_memory_ids(ids, 99)
    assert pos == 2
    assert out.tolist() == [[10, 11, 99, 12]]


def test_hook_writes_only_memory_index():
    ids = torch.tensor([[1, 2, 3]])
    new_ids, pos = insert_memory_ids(ids, 0)
    emb = nn.Embedding(8, 4)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[1:] = 1.0
    vec = torch.tensor([0.0, 0.0, 0.0, 4.0])

    def hook(_m, _i, out):
        h = out.clone()
        h[:, pos, :] = vec.to(dtype=h.dtype)
        return h

    handle = emb.register_forward_hook(hook)
    try:
        y = emb(new_ids)
    finally:
        handle.remove()
    assert float(y[0, pos, 3]) == 4.0
    assert float(y[0, -1].sum()) != 4.0 or float(y[0, -1, 3]) != 4.0
    assert pos != new_ids.shape[1] - 1
