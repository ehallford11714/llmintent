"""Prediction bind is at the mouth, not another residual write."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from llmintent.predictbind import (
    apply_member_bind,
    apply_state_gate,
    apply_unembed_bind,
    bayes_inform,
    bind_log_prior,
    bind_scores,
    lookup_token_id,
    member_surface,
    member_token_ids,
    predict_from_bind,
)


def test_lookup_refuses_first_piece_of_multitoken():
    class Tok:
        def encode(self, piece, add_special_tokens=False):
            if piece in ("12", " 12", "12."):
                return [1, 2]
            return [8]

    assert lookup_token_id(Tok(), "12") is None
    assert lookup_token_id(Tok(), "8") == 8


def test_member_surface_strips_kind():
    assert member_surface("digit:8") == "8"
    assert member_surface("noun:cat") == "cat"
    assert member_surface("12") == "12"


def test_member_bind_raises_only_named_ids():
    logits = torch.zeros(8)
    out = apply_member_bind(logits, [2, 5], gain=4.0)
    assert float(out[2]) == 4.0
    assert float(out[5]) == 4.0
    assert float(out[0]) == 0.0
    assert float(logits[2]) == 0.0


def test_unembed_bind_adds_projection():
    logits = torch.zeros(4)
    extra = torch.tensor([0.0, 1.0, 0.0, -0.5])
    out = apply_unembed_bind(logits, extra, gain=2.0)
    assert float(out[1]) == 2.0
    assert float(out[3]) == -1.0


def test_member_token_ids_are_unique():
    class Tok:
        def encode(self, piece, add_special_tokens=False):
            table = {"8": [8], " 8": [80], "2": [2], " 2": [20]}
            return table.get(piece, [99])

    bundle = SimpleNamespace(tokenizer=Tok())
    ids = member_token_ids(bundle, ["digit:8", "digit:2", "digit:8"])
    assert ids == [8, 2]


class _Head(nn.Module):
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(w)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F_linear(h, self.weight)


def F_linear(h, w):
    return torch.nn.functional.linear(h, w)


def test_predict_from_bind_members_moves_top():
    # last residual prefers token 0; members 1 and 2 get the lock bias.
    W = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ]
    )
    head = _Head(W)

    class Tok:
        def encode(self, piece, add_special_tokens=False):
            return {"8": [1], "2": [2]}.get(piece, [0])

        def decode(self, ids, skip_special_tokens=True):
            return {0: "is", 1: "8", 2: "2"}.get(int(ids[0]), "?")

    bundle = SimpleNamespace(
        tokenizer=Tok(),
        model=SimpleNamespace(lm_head=head),
    )
    hidden = torch.tensor([4.0, 0.0, 0.0])
    alone = predict_from_bind(bundle, hidden, members=(), mode="members", gain_members=0.0)
    bound = predict_from_bind(
        bundle,
        hidden,
        members=["digit:8", "digit:2"],
        mode="members",
        gain_members=8.0,
    )
    assert int(torch.argmax(alone)) == 0
    assert float(bound[1]) > float(alone[1])
    assert float(bound[2]) > float(alone[2])
    assert int(torch.argmax(bound)) in (1, 2)


def test_bayes_informs_when_uncertain_not_when_peaked():
    logits_peak = torch.tensor([8.0, 0.0, 0.0, 0.0])
    logits_flat = torch.zeros(4)
    log_pi = bind_log_prior(4, [1], member_mass=0.4)
    peak = bayes_inform(logits_peak, log_pi, max_strength=1.0)
    flat = bayes_inform(logits_flat, log_pi, max_strength=1.0)
    assert int(torch.argmax(peak)) == 0
    assert float(flat[1]) > float(flat[0])
    directed = apply_member_bind(logits_peak, [1], gain=8.0)
    assert float(peak[1] - peak[0]) < float(directed[1] - directed[0])


def test_bind_scores_bayes_does_not_add_gain():
    scores = torch.tensor([5.0, 0.0, 0.0])
    out = bind_scores(scores, member_ids=[1], mode="bayes", member_mass=0.02)
    assert int(torch.argmax(out)) == 0


def test_state_gate_flips_rival_to_live():
    scores = torch.tensor([0.0, 1.0, 4.0, 0.2])  # rival 2 is winning
    out, fired = apply_state_gate(scores, live_ids=[1], rival_ids=[2], boost=6.0)
    assert fired
    assert int(out.argmax()) == 1
    quiet, fired2 = apply_state_gate(scores, live_ids=[1], rival_ids=[2], boost=6.0)
    # wait, same scores would fire again - test non-state top
    quiet, fired2 = apply_state_gate(torch.tensor([9.0, 0.0, 0.0]), [1], [2], boost=6.0)
    assert not fired2
    assert int(quiet.argmax()) == 0


def test_bind_scores_members_only():
    scores = torch.tensor([1.0, 0.0, 0.5])
    out = bind_scores(scores, member_ids=[1], mode="members", gain_members=3.0)
    assert float(out[1]) == 3.0
    assert float(out[0]) == 1.0
    alone = bind_scores(scores, member_ids=[1], mode="alone", gain_members=3.0)
    assert float(alone[1]) == 0.0


def test_qual_verdict_improved_and_hijack():
    from llmintent.predictbind import assess, verdict

    item = {"expect": ["6"], "operands": ["8", "2"]}
    alone = assess("I need a calculator.", item)
    improved = assess("8 minus 2 equals 6 because you take two away.", item)
    hijack = assess("8 2", item)
    assert verdict(alone, improved) == "improved"
    assert verdict(alone, hijack) == "hijack"


def test_ablation_mine_requires_member_over_shuffle():
    from llmintent.predictbind import mine

    rows = [
        {
            "locked": True,
            "alone": {"track": 0.01, "pref": 0.0},
            "residual": {"track": 0.01, "pref": 0.0},
            "unembed": {"track": 0.01, "pref": 0.0},
            "members": {"track": 0.20, "pref": 0.18},
            "both": {"track": 0.21, "pref": 0.19},
            "shuffle_unembed": {"track": 0.01, "pref": 0.0},
            "shuffle_members": {"track": 0.02, "pref": 0.0},
        }
    ]
    s = mine(rows)
    assert s["rides_members"] and s["rides_both"]
    assert not s["rides_unembed"]
    rows[0]["members"]["track"] = 0.01
    rows[0]["both"]["track"] = 0.01
    assert not mine(rows)["rides_members"]
