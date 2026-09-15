"""Latent intent derivation and layer-to-layer tracking."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from llmintent.models import ModelBundle


def test_track_from_residuals_switches_intent():
    from llmintent.anatomy.intent_track import track_from_residuals

    rng = np.random.default_rng(7)
    dim = 32
    n = 6
    vis = rng.normal(size=dim)
    vis /= np.linalg.norm(vis)
    desc = rng.normal(size=dim)
    desc = desc - vis * np.dot(desc, vis)
    desc /= np.linalg.norm(desc)
    aud = rng.normal(size=dim)
    aud = aud - vis * np.dot(aud, vis) - desc * np.dot(aud, desc)
    aud /= np.linalg.norm(aud)

    mix = [
        vis * 3.0,
        vis * 3.0 + desc * 0.15,
        vis * 1.2 + desc * 1.4,
        desc * 3.0,
        desc * 3.0,
        desc * 3.0,
    ]
    prompt = np.stack(mix) + rng.normal(scale=0.01, size=(n, dim))
    probes = {
        "atlas.vision": np.stack([vis * 3.0] * n),
        "atlas.descending": np.stack([desc * 3.0] * n),
        "atlas.auditory": np.stack([aud * 3.0] * n),
    }
    track = track_from_residuals(
        prompt,
        probes,
        text="A dark shape is rushing toward me. Escape.",
        model_name="planted",
        probes={"atlas.vision": "v", "atlas.descending": "d", "atlas.auditory": "a"},
    )
    assert track.axis == "residual_layer"
    assert len(track.layers) == n
    assert track.layers[0].identified
    assert track.layers[0].top_intent == "atlas.vision"
    assert track.layers[-1].identified
    assert track.layers[-1].top_intent == "atlas.descending"
    kinds = {c.kind for c in track.changes}
    assert "switch" in kinds
    path = [x for i, x in enumerate(track.top_path) if i == 0 or x != track.top_path[i - 1]]
    assert path[0] == "atlas.vision"
    assert path[-1] == "atlas.descending"
    vis_span = next(s for s in track.spans if s.intent_id == "atlas.vision")
    desc_span = next(s for s in track.spans if s.intent_id == "atlas.descending")
    assert vis_span.onset == 0
    assert desc_span.peak >= vis_span.peak
    payload = track.to_dict()
    assert payload["any_identified"] is True
    md = track.to_markdown()
    assert "Latent intent through each layer" in md
    assert "Compile prior" in md


def test_track_prompt_compile_changes_across_spans():
    from llmintent.anatomy.intent_track import track_prompt_compile

    track = track_prompt_compile("I hear a song because a dark shape is looming. What should I do?")
    assert track.axis == "prompt_span"
    assert "vision" in track.compile_regions
    assert "auditory" in track.compile_regions
    assert len(track.layers) >= 2
    tops = {row.top_intent for row in track.layers if row.scores}
    assert any("auditory" in t or "atlas.auditory" in t for t in tops) or any(
        "atlas.vision" in t or "vision" in t for t in tops
    )
    assert track.changes


def test_select_probe_bank_includes_atlas_and_compile_hits():
    from llmintent.anatomy.intent_track import select_probe_bank

    bank = select_probe_bank("A dark shape is rushing toward me. Stop.", max_probes=22)
    assert "atlas.vision" in bank
    assert "atlas.descending" in bank or "fn.stop_action" in bank or "fn.looming_approach" in bank
    assert len(bank) <= 22


def test_tiny_model_tracks_every_layer():
    from llmintent.anatomy.intent_track import track_latent_intent

    class Tok:
        chat_template = None

        def __call__(self, text, return_tensors="pt"):
            ids = [2 + (ord(c) % 28) for c in (text or "x")[:16]] or [2]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

        def decode(self, ids, skip_special_tokens=True):
            return "x"

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8, vocab_size=32)
            self.embed = nn.Embedding(32, 8)
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
            self.lm_head = nn.Linear(8, 32, bias=False)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True, use_cache=False, **kw):
            h = self.embed(input_ids)
            states = [h]
            for layer in self.layers:
                h = torch.tanh(layer(h))
                states.append(h)
            return SimpleNamespace(logits=self.lm_head(h), hidden_states=tuple(states))

    bundle = ModelBundle(
        "tiny-track",
        tokenizer=Tok(),
        model=Tiny(),
        device=torch.device("cpu"),
        is_causal=True,
        num_layers=4,
    )
    track = track_latent_intent(
        bundle,
        "A dark shape is rushing toward me.",
        max_probes=6,
        probes={
            "atlas.vision": "A dark shape is rushing toward me in the light.",
            "atlas.auditory": "I hear a buzzing pulse song in the air.",
            "inquire": "What is happening, and why is it happening?",
        },
    )
    assert track.axis == "residual_layer"
    assert len(track.layers) == 5  # embed + 4 blocks
    assert len(track.changes) == 4
    assert set(track.probes) >= {"atlas.vision"}
    rows = track.as_layer_intents()
    assert len(rows) == 5
    assert "atlas.vision" in rows[0].intents
    thoughts = track.as_layer_thoughts()
    assert thoughts[0].layer == 0


def test_trajectory_uses_measured_track(monkeypatch):
    from llmintent.anatomy import trajectory as run_traj
    from llmintent.anatomy.intent_track import track_from_residuals
    from llmintent.anatomy.trajectory import AnatomyTrajectory

    rng = np.random.default_rng(1)
    vis = rng.normal(size=16)
    vis /= np.linalg.norm(vis)
    n = 4
    prompt = np.stack([vis * 3] * n)
    probes = {"atlas.vision": np.stack([vis * 3] * n), "atlas.auditory": rng.normal(size=(n, 16))}
    planted = track_from_residuals(prompt, probes, text="a dark shape looming", model_name="planted")

    def _fake_track(bundle, text, **kwargs):
        return planted

    monkeypatch.setattr("llmintent.anatomy.intent_track.track_latent_intent", _fake_track)
    traj = run_traj(
        "a dark shape looming",
        bundle=SimpleNamespace(name="planted"),
        print_flag=False,
        measure_latent=True,
    )
    assert isinstance(traj, AnatomyTrajectory)
    assert traj.intent_track is not None
    assert traj.residual_identified is True
    assert traj.layers[0].intents["atlas.vision"] > traj.layers[0].intents.get("atlas.auditory", 0)
    md = traj.to_markdown()
    assert "Latent intent through each layer" in md


def test_battery_offline_compile_spans():
    from llmintent.anatomy.intent_battery import run_intent_battery

    report = run_intent_battery(None, cases=(
        {"id": "a", "label": "A", "text": "I hear a song.", "looks_for": "auditory"},
        {"id": "b", "label": "B", "text": "A dark shape is rushing toward me.", "looks_for": "vision"},
    ))
    assert report["n_cases"] == 2
    assert report["cases"][0]["axis"] == "prompt_span"
    assert "auditory" in report["cases"][0]["surface_compile"] or "atlas.auditory" in str(report["cases"][0])
    assert "vision" in report["cases"][1]["surface_compile"]


def test_cli_intent_track_no_model():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [sys.executable, "-m", "llmintent", "intent-track", "--text", "I hear a song.", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"axis": "prompt_span"' in proc.stdout
    assert "intent_track" in proc.stdout
