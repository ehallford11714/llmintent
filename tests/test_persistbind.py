"""Frozen-decay spike bank holds bindings only while traces last."""

from __future__ import annotations

import torch

from llmintent.persistbind import PersistenceBinder, drive_from_hidden
from llmintent.spike import SpikeConfig, SpikePotentialBank


def test_decay_is_not_a_parameter():
    bank = SpikePotentialBank(8, SpikeConfig(decay=0.9))
    assert list(bank.parameters()) == []
    assert list(bank.named_parameters()) == []
    assert bank.decay == 0.9
    bank.train(True)
    assert not bank.training
    try:
        torch.optim.SGD(list(bank.parameters()), lr=1.0)
        raise AssertionError("SGD should refuse an empty parameter list")
    except ValueError as exc:
        assert "empty parameter list" in str(exc)


def test_silent_steps_leak_trace():
    bank = SpikePotentialBank(2, SpikeConfig(decay=0.5, threshold=0.5))
    spikes = bank.step(torch.tensor([2.0, 0.0]))
    assert float(spikes[0]) == 1.0
    held = float(bank.trace[0])
    bank.idle(3)
    # trace <- 0.5 * trace each idle tick, no new spike
    assert float(bank.trace[0]) == held * (0.5**3)
    assert float(bank.v[0]) < 1e-6


def test_coincidence_writes_binding_then_idle_expires_it():
    cfg = SpikeConfig(
        decay=0.5,
        threshold=0.5,
        bind_floor=0.05,
        live_floor=0.15,
        max_nodes=4,
        fire_k=10.0,  # residual channels stay quiet; only named cells spike
    )
    binder = PersistenceBinder(cfg)
    # Two tokens, each a distinct entity, then silence.
    h0 = torch.zeros(4)
    h1 = torch.zeros(4)
    binder.begin_token()
    binder.accumulate_layer(h0)
    binder.end_token(token="8", residual=h0, entity_name="digit:8")
    binder.begin_token()
    binder.accumulate_layer(h1)
    binder.end_token(token="2", residual=h1, entity_name="digit:2")
    live = binder.live_bindings()
    names = {(b.left_name, b.right_name) for b in live}
    assert ("digit:8", "digit:2") in names or ("digit:2", "digit:8") in names
    binder.idle(8)
    assert binder.live_bindings() == []
    assert any("digit:8" in e and "digit:2" in e for e in binder.expired)


def test_respike_refreshes_expired_binding():
    cfg = SpikeConfig(
        decay=0.5,
        threshold=0.5,
        bind_floor=0.05,
        live_floor=0.15,
        max_nodes=4,
        fire_k=10.0,
    )
    binder = PersistenceBinder(cfg)
    z = torch.zeros(4)
    binder.begin_token()
    binder.accumulate_layer(z)
    binder.end_token(token="8", residual=z, entity_name="digit:8")
    binder.begin_token()
    binder.accumulate_layer(z)
    binder.end_token(token="2", residual=z, entity_name="digit:2")
    binder.idle(8)
    assert binder.live_bindings() == []
    binder.begin_token()
    binder.accumulate_layer(z)
    binder.end_token(token="8", residual=z, entity_name="digit:8")
    binder.begin_token()
    binder.accumulate_layer(z)
    binder.end_token(token="2", residual=z, entity_name="digit:2")
    live = binder.live_bindings()
    assert live
    assert live[0].potential >= 0.15


def test_drive_from_hidden_token_clock():
    cfg = SpikeConfig(decay=0.8, threshold=0.5, max_nodes=8, fire_k=10.0)
    binder = PersistenceBinder(cfg)
    # 2 layers, 5 tokens, dim 8. Entities at positions 0 and 2.
    H = torch.zeros(2, 5, 8)
    drive_from_hidden(
        binder,
        [H[0:1], H[1:2]],
        tokens=["8", "minus", "2", "equals", "is"],
        spans=[
            {"position": 0, "token": "8", "kind": "digit"},
            {"position": 2, "token": "2", "kind": "digit"},
        ],
    )
    assert binder.bank is not None
    assert binder.bank.clock == 5
    traces = binder.named_traces()
    assert traces["digit:8"] < traces["digit:2"]  # older mention leaked more
    assert traces["digit:2"] > 0


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def test_bound_gradient_is_birth_state_not_last_token():
    cfg = SpikeConfig(decay=0.8, threshold=0.5, max_nodes=4, fire_k=10.0, live_floor=0.05)
    binder = PersistenceBinder(cfg)
    e8 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    last = torch.tensor([0.0, 0.0, 1.0, 0.0])
    steps = [("8", e8, "digit:8"), ("minus", last, None), ("2", e2, "digit:2"), ("is", last, None)]
    for tok, h, name in steps:
        binder.begin_token()
        binder.accumulate_layer(h)
        binder.end_token(token=tok, residual=h, entity_name=name)
    vec = binder.state_vector()
    assert vec is not None
    mix = (e8 + e2) / 2
    assert _cos(vec, mix) > 0.7
    assert _cos(vec, last) < 0.2
    eight_idx = binder.name_to_cell["digit:8"]
    torch.testing.assert_close(binder.payload[eight_idx], e8)


def test_bound_state_locks_and_idle_cannot_unwrite():
    cfg = SpikeConfig(decay=0.8, threshold=0.5, max_nodes=4, fire_k=10.0, live_floor=0.02)
    binder = PersistenceBinder(cfg)
    e8 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    for tok, h, name in [("8", e8, "digit:8"), ("2", e2, "digit:2")]:
        binder.begin_token()
        binder.accumulate_layer(h)
        binder.end_token(token=tok, residual=h, entity_name=name)
    assert binder.bound_state is not None
    assert binder.bound_state.locked
    assert set(binder.bound_state.members) == {"digit:8", "digit:2"}
    before = binder.bound_state.vector.clone()
    scale0 = binder.bound_scale
    binder.idle(8)
    torch.testing.assert_close(binder.bound_state.vector, before)
    assert binder.bound_scale == scale0
    assert binder.state_vector() is not None
    last = torch.tensor([0.0, 0.0, 1.0, 0.0])
    binder.begin_token()
    binder.accumulate_layer(last)
    binder.end_token(token="is", residual=last, entity_name=None)
    torch.testing.assert_close(binder.bound_state.vector, before)
    assert _cos(binder.state_vector(), last) < 0.2


def test_third_member_grows_locked_state():
    cfg = SpikeConfig(decay=0.8, threshold=0.5, max_nodes=8, fire_k=10.0, live_floor=0.02)
    binder = PersistenceBinder(cfg)
    e8 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    e3 = torch.tensor([0.0, 0.0, 1.0, 0.0])
    for tok, h, name in [("8", e8, "digit:8"), ("2", e2, "digit:2")]:
        binder.begin_token()
        binder.accumulate_layer(h)
        binder.end_token(token=tok, residual=h, entity_name=name)
    assert binder.bound_state is not None and binder.bound_state.locked
    before = binder.bound_state.vector.clone()
    binder.begin_token()
    binder.accumulate_layer(e3)
    binder.end_token(token="3", residual=e3, entity_name="digit:3")
    assert set(binder.bound_state.members) == {"digit:2", "digit:3", "digit:8"}
    assert _cos(binder.bound_state.vector, before) < 0.99


def test_two_mentions_same_name_locks():
    cfg = SpikeConfig(decay=0.8, threshold=0.5, max_nodes=4, fire_k=10.0, live_floor=0.02)
    binder = PersistenceBinder(cfg)
    e6 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(2):
        binder.begin_token()
        binder.accumulate_layer(e6)
        binder.end_token(token="6", residual=e6, entity_name="digit:6")
    assert binder.mentions == 2
    assert binder.bound_state is not None and binder.bound_state.locked


def test_bind_in_adds_locked_state_not_last_token():
    cfg = SpikeConfig(decay=0.8, threshold=0.5, max_nodes=4, fire_k=10.0)
    binder = PersistenceBinder(cfg)
    e8 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    last = torch.tensor([0.0, 0.0, 1.0, 0.0])
    for tok, h, name in [("8", e8, "digit:8"), ("2", e2, "digit:2")]:
        binder.begin_token()
        binder.accumulate_layer(h)
        binder.end_token(token=tok, residual=h, entity_name=name)
    mixed = binder.bind_in(last, gain=2.0)
    assert _cos(mixed, (e8 + e2) / 2) > _cos(last, (e8 + e2) / 2)
    assert binder.bound_state is not None and binder.bound_state.locked


def test_ablation_mine_requires_bound_over_shuffle():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from bound_state_ablation import mine

    rows = [
        {
            "locked": True,
            "alone": {"track": 0.01, "pref": 0.0},
            "bound": {"track": 0.08, "pref": 0.06},
            "ablate_random": {"track": 0.02, "pref": 0.0},
            "ablate_shuffle": {"track": 0.02, "pref": 0.0},
            "bound_mid": {"track": 0.07, "pref": 0.05},
        }
    ]
    s = mine(rows)
    assert s["rides_bind"]
    rows[0]["bound"]["track"] = 0.01
    assert not mine(rows)["rides_bind"]
