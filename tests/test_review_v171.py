"""Regression checks for the v1.7.0 review (P1/P2)."""

from __future__ import annotations

import numpy as np
import pytest


def test_alignment_uses_action_score_not_correctness_margin():
    from llmintent.anatomy.fly_assay import run_looming_giant_fibre_assay
    from llmintent.anatomy.region_test import (
        align_vectors,
        fly_condition_vector,
        llm_condition_vector,
        score_synthetic_alignment,
        synthetic_llm_rows,
    )
    from llmintent.anatomy.tasks import by_split

    assay = run_looming_giant_fibre_assay()
    items = by_split("looming_language", "confirmation")
    fly = fly_condition_vector(assay)

    matched = score_synthetic_alignment(assay, items, mode="matched")
    assert matched["alignment"] is not None and matched["alignment"] > 0.5

    rows = synthetic_llm_rows(items, mode="matched")
    assert all(r["margin"] == 1.0 for r in rows)
    margin_only = [{**r, "action_score": r["margin"]} for r in rows]
    align_margin, _ = align_vectors(fly, llm_condition_vector(margin_only))
    assert align_margin is None or abs(align_margin) < 0.2

    wrong = synthetic_llm_rows(items, mode="incorrect_recede")
    margin_wrong = [{**r, "action_score": r["margin"]} for r in wrong]
    align_bug, _ = align_vectors(fly, llm_condition_vector(margin_wrong))
    assert align_bug is not None and align_bug > 0.5
    align_action, _ = align_vectors(fly, llm_condition_vector(wrong))
    assert align_action is not None and align_action > 0.5


def test_oriented_action_score_treats_wait_as_nonaction():
    from llmintent.anatomy.tasks import all_items, oriented_action_score

    recede = next(it for it in all_items() if "receding" in it.tags)
    scored = {c: 2.0 for c in recede.positive}
    scored.update({c: 0.0 for c in recede.negative})
    assert oriented_action_score(recede, scored) < 0
    margin = max(scored[c] for c in recede.positive) - max(scored[c] for c in recede.negative)
    assert margin > 0


def test_cascade_verdict_requires_a_changes_b_and_rescue():
    from llmintent.anatomy.cascade import cascade_verdict
    from llmintent.anatomy.evidence import UNIDENTIFIED

    assert (
        cascade_verdict(
            a_changes_b=True,
            task_disrupted=True,
            task_restored=True,
            vs_random=True,
        )
        == "intervention-supported pathway"
    )
    assert (
        cascade_verdict(
            a_changes_b=False,
            task_disrupted=True,
            task_restored=True,
            vs_random=True,
        )
        == UNIDENTIFIED
    )
    assert (
        cascade_verdict(
            a_changes_b=True,
            task_disrupted=True,
            task_restored=False,
            vs_random=True,
        )
        == "candidate cascade"
    )


def test_packed_quant_storage_is_rejected():
    from llmintent.anatomy.weights import WeightExtractionError, extract_dense_weight

    class Packed:
        quant_state = object()
        shape = (64,)
        dtype = "uint8"

        def float(self):
            return np.arange(64, dtype=np.float32)

        def numpy(self):
            return np.arange(64, dtype=np.float32)

    with pytest.raises(WeightExtractionError):
        extract_dense_weight(Packed())

    dense = np.arange(12, dtype=np.float64).reshape(3, 4)
    got = extract_dense_weight(dense, expected_shape=(3, 4))
    assert got.shape == (3, 4)
    assert np.allclose(got, dense)


def test_dense_weight_roundtrip_matches_float_cast():
    torch = pytest.importorskip("torch")
    from llmintent.anatomy.weights import extract_dense_weight

    weight = torch.randn(8, 16)
    extracted = extract_dense_weight(weight, expected_shape=(8, 16))
    assert np.allclose(extracted, weight.float().cpu().numpy(), atol=1e-6)


def test_unidentified_tied_prototypes_do_not_flag_deception():
    from llmintent.anatomy.intent_track import track_from_residuals
    from llmintent.anatomy.misalign import scan_negative_intent

    rng = np.random.default_rng(0)
    prompt = rng.normal(size=(3, 8))
    probes = {
        "atlas.vision": prompt.copy(),
        "deception": prompt.copy(),
        "inquire": prompt.copy(),
    }
    track = track_from_residuals(prompt, probes, text="What is two plus two?")
    assert track.any_identified is False
    rows = [row.to_dict() for row in track.as_layer_intents()]
    assert all(not row.get("identified") for row in rows)
    assert all(row["intents"].get("deception", 0.0) == 0.0 for row in rows)
    _, flag = scan_negative_intent("What is two plus two?", layer_intents=rows)
    assert not any(loc.intent == "deception" and loc.kind.startswith("layer.") for loc in flag.loci)


def test_cosine_rejects_dim_mismatch():
    from llmintent.anatomy.intent_track import _cos

    with pytest.raises(ValueError, match="dim mismatch"):
        _cos(np.ones(4), np.ones(3))


def test_confirmation_template_groups_are_held_out():
    from llmintent.anatomy.tasks import by_split, holdout_items

    disc = {it.template_group for it in by_split("looming_language", "discovery")}
    conf = {it.template_group for it in holdout_items("looming_language")}
    assert conf
    assert disc.isdisjoint(conf)
    val_disc = {it.template_group for it in by_split("value_modulation", "discovery")}
    val_conf = {it.template_group for it in holdout_items("value_modulation")}
    assert val_conf
    assert val_disc.isdisjoint(val_conf)


def test_fly_assay_does_not_claim_biological_validation():
    from llmintent.anatomy.fly_assay import (
        run_dopamine_malecns_assay,
        run_looming_giant_fibre_assay,
        run_looming_malecns_assay,
    )
    from llmintent.anatomy.region_test import validate_analogous_region_test

    for assay in (
        run_looming_giant_fibre_assay(),
        run_looming_malecns_assay(),
        run_dopamine_malecns_assay(),
    ):
        payload = assay.to_dict()
        assert payload["biological_evidence"] == "not_established"
        assert payload["validates_biology"] is False
        assert payload["simulator_mechanics"] == assay.validation_status

    report = validate_analogous_region_test()
    assert report["biological_evidence"] == "not_established"
    assert report["alignment_uses"] == "action_score"
    assert report["mechanics_pass"] is True


def test_unavailable_logit_profile_is_not_ranked():
    from llmintent.anatomy.spaces import signed_logit_profile

    profile = signed_logit_profile(
        np.ones(4),
        np.ones((10, 7)),
        tokenizer=None,
    )
    assert profile.get("source") == "unavailable"


def test_predictbind_helpers_live_in_package():
    from llmintent.predictbind import assess, mine, verdict

    item = {"expect": ["6"], "operands": ["8", "2"]}
    alone = assess("I need a calculator.", item)
    improved = assess("8 minus 2 equals 6 because you take two away.", item)
    assert verdict(alone, improved) == "improved"
    summary = mine(
        [
            {
                "locked": True,
                "alone": {"track": 0.01, "pref": 0.0},
                "members": {"track": 0.2, "pref": 0.1},
                "both": {"track": 0.2, "pref": 0.1},
                "unembed": {"track": 0.01, "pref": 0.0},
                "shuffle_unembed": {"track": 0.01, "pref": 0.0},
                "shuffle_members": {"track": 0.02, "pref": 0.0},
            }
        ]
    )
    assert summary["rides_members"]
