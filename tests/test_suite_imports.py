"""Offline suite import + API smoke tests (no HF downloads, no torch forward)."""

from __future__ import annotations

import json
import subprocess
import sys


def test_isolates_identify_offline():
    from llmintent.isolates import form_motifs, identify_isolates, trajectory_from_motifs

    text = (
        "I want to finish the report. I cannot miss the deadline. "
        "I feel anxious. I will submit it using the portal so that it is on time."
    )
    isos = identify_isolates(text=text)
    assert len(isos) >= 2
    motifs = form_motifs(isos)
    assert isinstance(motifs, list)
    traj = trajectory_from_motifs(motifs, isos)
    assert traj.steps
    assert traj.ascii_diagram


def test_motifs_alias():
    from llmintent.motifs import form_motifs, trajectory_from_motifs
    from llmintent.isolates import identify_isolates

    isos = identify_isolates(text="I want X but cannot Y")
    motifs = form_motifs(isos)
    traj = trajectory_from_motifs(motifs, isos)
    assert traj.layer_path


def test_iv_motifs_offline():
    from llmintent.iv_motifs import LayerCausalSuite
    from llmintent.causal_layers import LayerCausalSuite as Alias

    assert Alias is LayerCausalSuite
    suite = LayerCausalSuite.from_text(
        "I want to finish. I feel stuck. I will submit so that it is done."
    )
    result = suite.run(outcome_hint="done", mock_iv=True, n_bootstrap=24, seed=3)
    assert result.isolates
    assert result.motifs is not None
    md = result.to_markdown()
    assert "Indication" in md or "indication" in md.lower()
    d = result.to_dict()
    assert "indication_by_layer" in d


def test_latent_inspect_offline():
    from llmintent import latent

    info = latent.describe()
    assert info["available"] is True
    assert latent.available() is True
    assert latent.backend_name()
    report = latent.inspect_text(
        "I want a refund but cannot wait. What should I do?",
        backend="rule",
        include_sae=True,
        include_probe_train=True,
    )
    d = report.to_dict()
    assert d["hypothesized_intents"]
    assert d["caveats"]
    assert "disclaimer" in d
    assert report.summary_lines()


def test_cli_latent():
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmintent",
            "latent",
            "--text",
            "Please explain gravity step by step.",
            "--no-sae",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["backend"] == "rule"
    assert data["hypothesized_intents"]


def test_backend_source_documented():
    from llmintent import isolates

    src = isolates.backend_source
    assert src in ("vendored", "intentisolates")


def test_cli_isolates_motifs_iv():
    text = "I want coffee. I cannot wait. I will brew it so that I wake up."
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONIOENCODING": "utf-8"}
    cases = [
        ([sys.executable, "-m", "llmintent", "isolates", "--text", text], True),
        ([sys.executable, "-m", "llmintent", "motifs", "--text", text], True),
        ([sys.executable, "-m", "llmintent", "reasoning-trajectory", "--text", text], True),
        (
            [
                sys.executable,
                "-m",
                "llmintent",
                "iv-motifs",
                "--text",
                text,
                "--outcome-hint",
                "wake",
                "--mock-iv",
                "--format",
                "json",
            ],
            True,
        ),
        ([sys.executable, "-m", "llmintent", "trajectory", "--text", text], True),
        ([sys.executable, "-m", "llmintent", "models", "list", "--family", "legacy"], True),
    ]
    for argv, expect_json in cases:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False, env=env)
        assert proc.returncode == 0, (argv, proc.stderr, proc.stdout[:500])
        if expect_json and "models" not in argv:
            json.loads(proc.stdout)
