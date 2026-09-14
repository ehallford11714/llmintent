"""Offline anatomy: compile, connectome IV, SVD mapping, A vs B ablation."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np


def test_atlas_regions_declare_handles_and_wiring():
    from llmintent.anatomy import default_atlas, region_ids

    atlas = default_atlas()
    ids = set(region_ids())
    assert "vision" in ids and "causal_logic" in ids and "auditory" in ids
    for r in atlas.regions:
        assert r.handles
        assert r.fly_neuropil
        assert r.band in ("sensory", "central", "motor")
        assert r.id in atlas.integrates
    # Vision integrates with descending (giant fibre) and/or workspace
    vision_nb = set(atlas.integrates["vision"])
    assert vision_nb & {"descending", "workspace", "causal_logic"}


def test_compile_maps_intent_not_only_alias():
    from llmintent.anatomy import compile_regions

    looming = compile_regions("a dark shape rushing toward me")
    assert "vision" in looming.regions

    hungry = compile_regions("i am hungry")
    assert "gustatory" in hungry.regions

    causal = compile_regions("if the light is on then turn left because that causes escape")
    assert "causal_logic" in causal.regions
    assert "vision" in causal.regions

    mixed = compile_regions("I hear a song because a dark shape is looming.")
    assert "auditory" in mixed.regions
    assert "vision" in mixed.regions
    assert "causal_logic" in mixed.regions

    fiction = compile_regions("Once upon a time a little girl lived in a village.")
    assert "vision" not in fiction.regions
    assert fiction.dropped or not fiction.hits


def test_connectome_flags_exclusion_violation():
    from llmintent.anatomy import literature_region_connectome

    conn = literature_region_connectome()
    viol = {(e.source, e.target) for e in conn.exclusion_violations()}
    assert ("vision", "descending") in viol
    assert conn.has_path("olfactory", "associative")
    assert "olfactory" in conn.valid_instruments("associative")


def test_svd_recovers_planted_region_axes():
    from llmintent.anatomy import map_synthetic

    dim = 24
    v = np.eye(dim)[0]
    a = np.eye(dim)[1]
    H = np.vstack([np.tile(v * 4.0, (6, 1)), np.tile(a * 4.0, (6, 1))])
    anatomy = map_synthetic(
        H,
        {"vision": v, "auditory": a},
        layer_index=list(range(12)),
    )
    assert anatomy.occupancy["vision"] > 0.25
    assert anatomy.occupancy["auditory"] > 0.25
    assert set(anatomy.layer_region.values()) <= {"vision", "auditory"}


def test_ablation_a_vs_b_changes_output():
    from llmintent.anatomy import plant_and_ablate

    result, axes = plant_and_ablate("vision", "auditory", gain=5.0, seed=1)
    assert result.changed
    assert result.top_a[0] != result.top_b[0]
    assert result.kl_ab > 0.5
    assert result.handles_a and result.handles_b
    assert "vision" in axes and "auditory" in axes


def test_connectome_iv_uses_sensory_instruments():
    from llmintent.anatomy import iv_from_text

    plan, iv = iv_from_text(
        "I see a dark shape. I hear a song. I will escape because it is looming.",
        mock_iv=True,
        seed=3,
    )
    assert plan.hits
    assert iv.instruments_used
    assert all(
        z in ("vision", "auditory", "olfactory", "gustatory", "somatosensory")
        for z in iv.instruments_used
    )
    assert iv.exclusion_violations
    d = iv.to_dict()
    assert "causation_edges" in d


def test_map_anatomy_offline_report():
    from llmintent.anatomy import map_anatomy

    report = map_anatomy(
        "A dark buzzing shape is rushing toward me so I should escape.",
        region_a="vision",
        region_b="auditory",
        mock_iv=True,
        seed=5,
    )
    assert report.cards
    vision = next(c for c in report.cards if c.id == "vision")
    assert vision.handles
    assert vision.integrates_with
    assert report.ablation is not None
    assert report.ablation.changed
    md = report.to_markdown()
    assert "Handles" in md
    assert "Ablation" in md
    payload = report.to_dict()
    assert payload["ablation"]["changed"] is True


def test_cli_compile_and_anatomy():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONIOENCODING": "utf-8"}
    env["PYTHONPATH"] = str(root / "src") + (__import__("os").pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    compile_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmintent",
            "compile",
            "--text",
            "a dark shape rushing toward me",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr
    data = json.loads(compile_proc.stdout)
    assert "vision" in data["regions"]

    anat = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmintent",
            "anatomy",
            "--text",
            "I hear a song because a dark shape is looming.",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert anat.returncode == 0, anat.stderr
    payload = json.loads(anat.stdout)
    assert payload["regions"]
    assert payload["ablation"]["changed"] is True


def test_latent_thought_report_maps_to_thought_report():
    from llmintent.anatomy.thoughts import LatentThoughtReport, LayerThought
    from llmintent.latent_vendor.types import IntentHypothesis, LayerSaliency

    lat = LatentThoughtReport(
        text="I hear a song because a dark shape is looming.",
        model_name="Qwen/Qwen3.8-27B",
        compiled_regions=["auditory", "causal_logic", "vision"],
        thoughts=[
            LayerThought(
                layer=12,
                depth=0.2,
                band="sensory",
                top_tokens=["hear", "song"],
                region="auditory",
                region_score=0.4,
                residual_l2=1.2,
            )
        ],
        occupancy={"auditory": 0.6, "vision": 0.4},
        hypothesized_intents=[
            IntentHypothesis(tag="auditory", score=0.4, method="logit_lens_region")
        ],
        layer_saliency=[LayerSaliency(layer=12, score=1.2, source="residual_l2_last_token")],
        notes=["unit"],
    )
    report = lat.to_thought_report()
    d = report.to_dict()
    assert d["backend"] == "hf"
    assert d["logit_lens"][0]["region"] == "auditory"
    assert "disclaimer" in d
    assert d["metadata"]["compiled_regions"] == ["auditory", "causal_logic", "vision"]
