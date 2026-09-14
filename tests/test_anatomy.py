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
    assert vision.does
    assert vision.varies
    assert vision.series
    assert vision.integrates_with
    assert report.ablation is not None
    assert report.ablation.changed
    assert report.trace is not None
    md = report.to_markdown()
    assert "Does" in md
    assert "Through the prompt" in md
    assert "Ablation" in md
    payload = report.to_dict()
    assert payload["ablation"]["changed"] is True
    assert payload["trace"]["spans"]


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
    assert any(r.get("does") for r in payload["regions"])
    assert payload["trace"]["spans"]


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


def test_what_region_does():
    from llmintent.anatomy import what_region_does

    vision = what_region_does("vision")
    assert "looming" in vision.lower() or "luminance" in vision.lower()
    assert "Band:" in vision
    assert "Fly analogue:" in vision


def test_trace_prompt_varies_through_spans():
    from llmintent.anatomy import trace_prompt

    trace = trace_prompt("I hear a song because a dark shape is looming.")
    assert len(trace.spans) >= 2
    vision = next(r for r in trace.regions if r.id == "vision")
    auditory = next(r for r in trace.regions if r.id == "auditory")
    assert max(vision.series) > 0
    assert max(auditory.series) > 0
    assert vision.series != auditory.series
    assert vision.peak_span != auditory.peak_span
    assert vision.varies
    assert auditory.varies
    assert "silent" not in vision.varies.lower() or vision.peak_span is not None


def test_template_and_agent_draft(monkeypatch):
    from llmintent.anatomy import draft_anatomy_report, map_anatomy

    monkeypatch.delenv("LLMINTENT_GUIDE_URL", raising=False)
    monkeypatch.delenv("LLMINTENT_GUIDE_KEY", raising=False)
    report = map_anatomy(
        "I hear a song because a dark shape is looming.",
        draft=True,
        mock_iv=True,
        seed=5,
    )
    assert report.draft
    lowered = report.draft.lower()
    assert "vision" in lowered
    assert "auditory" in lowered
    assert "because" in report.draft.lower() or "causal" in lowered

    guided = draft_anatomy_report(
        report, agent=lambda system, user: "CUSTOM AGENT DRAFT for vision"
    )
    assert guided.backend == "agent"
    assert "CUSTOM AGENT DRAFT" in guided.markdown


def test_anatomy_offline_graph_and_layers():
    from llmintent.anatomy import Anatomy, intent_ids

    anat = Anatomy.offline("I hear a song because a dark shape is looming.", n_layers=8)
    assert anat.n_layers == 8
    assert len(anat.layers) == 8
    ids = set(intent_ids())
    assert ids <= set(anat.layers[0].intents)
    assert anat.graph.nodes
    assert anat.graph.edges
    md = anat.to_markdown()
    assert "Responsible for" in md
    assert "flowchart" in anat.graph.mermaid()
    payload = anat.to_dict()
    assert payload["name"] == "Anatomy"
    assert payload["graph"]["n_edges"] > 0


def test_trajectory_all_layers_all_intents():
    from llmintent.anatomy import trajectory
    from llmintent.anatomy.thoughts import LayerThought

    thoughts = [
        LayerThought(
            layer=i,
            depth=i / 7,
            band="sensory" if i < 3 else ("central" if i < 6 else "motor"),
            top_tokens=["song"] if i == 0 else (["危险", "威胁"] if i == 4 else ["."]),
            region="workspace",
            region_score=0.05,
            residual_l2=1.0,
        )
        for i in range(8)
    ]
    traj = trajectory(
        "I hear a song because a dark shape is looming.",
        thoughts=thoughts,
        print_flag=False,
        all_layers=True,
        n_layers=8,
    )
    assert traj.method == "trajectory"
    assert len(traj.layers) == 8
    assert traj.intent_ids
    for row in traj.layers:
        assert set(row.intents) == set(traj.intent_ids)
    assert any(row.active for row in traj.layers)
    md = traj.to_markdown()
    assert "All intents through each layer" in md


def test_misalign_flag_prints_on_harm(capsys):
    from llmintent.anatomy.misalign import FLAG_NAME, notify_misalign, scan_negative_intent

    _loci, flag = scan_negative_intent("I will attack him and hide this from everyone.")
    assert flag.triggered
    notify_misalign(flag)
    captured = capsys.readouterr()
    assert FLAG_NAME in captured.err
    assert "MisAlign Flag" in flag.banner()


def test_map_anatomy_includes_anatomy_graph():
    from llmintent.anatomy import map_anatomy

    report = map_anatomy("I hear a song because a dark shape is looming.", mock_iv=True, seed=3)
    assert report.anatomy is not None
    assert report.anatomy.graph.edges
    md = report.to_markdown()
    assert "layer responsibility" in md.lower() or "Responsible for" in md
    assert report.to_dict()["anatomy"]["n_layers"] >= 1

