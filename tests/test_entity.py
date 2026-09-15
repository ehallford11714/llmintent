"""Locked entity slots persist birth residuals after the last token moves."""

from __future__ import annotations

import numpy as np
import torch


def test_write_slots_do_not_follow_last_token():
    from llmintent.entity import write_slots

    # 2 layers, 5 positions, dim 8. Position 0 is the entity; last is noise.
    h0 = torch.zeros(1, 5, 8)
    h1 = torch.zeros(1, 5, 8)
    h1[0, 0] = torch.arange(8, dtype=torch.float32)
    h1[0, -1] = torch.ones(8)
    slots = write_slots(
        [h0, h1],
        [{"id": "digit:8@0", "kind": "digit", "token": "8", "position": 0}],
        layer=-1,
    )
    assert len(slots) == 1
    assert slots[0].locked
    np.testing.assert_allclose(slots[0].vector, np.arange(8, dtype=np.float64))
    assert not np.allclose(slots[0].vector, np.ones(8))


def test_detect_merges_consecutive_digits():
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            table = {0: "1", 1: "6", 2: " minus", 3: " 2"}
            return table[int(ids[0])]

    from llmintent.entity import detect_entity_spans

    ids = torch.tensor([[0, 1, 2, 3]])
    spans = detect_entity_spans(Tok(), ids)
    tokens = [s["token"] for s in spans]
    assert "16" in tokens
    assert "2" in tokens
    sixteen = next(s for s in spans if s["token"] == "16")
    assert sixteen["position"] == 1


def test_locked_slot_still_matches_entity_when_last_does_not():
    from llmintent.entity import write_slots

    dim = 4
    entity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    last = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    H = torch.zeros(1, 3, dim)
    H[0, 0] = torch.tensor(entity)
    H[0, 2] = torch.tensor(last)
    slots = write_slots(
        [H],
        [{"id": "digit:8@0", "kind": "digit", "token": "8", "position": 0}],
    )
    slot = slots[0].vector
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cos(slot, entity) > 0.99
    assert cos(last, entity) < 0.1
    assert cos(slot, entity) > cos(last, entity) + 0.5
