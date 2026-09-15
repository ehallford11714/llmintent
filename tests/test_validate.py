"""Kill-or-keep anatomy validation (offline)."""

from __future__ import annotations


def test_offline_validation_compile_survives():
    from llmintent.anatomy.validate import run_validation

    report = run_validation()
    by_id = {r.id: r for r in report.results}
    assert by_id["compile_gold"].verdict == "keep"
    assert by_id["compile_drop"].verdict == "keep"
    assert by_id["span_varies"].verdict == "keep"
    assert by_id["connectome_iv"].verdict == "keep"
    assert by_id["planted_svd"].verdict == "keep"
    assert by_id["planted_ablation"].verdict == "keep"
    assert by_id["weight_shuffle"].skipped
    assert by_id["causal_steer"].skipped
    assert "compile_catalogue" in report.next_version
    assert "fly_is_the_llm" in report.next_version
    md = report.to_markdown()
    assert "Anatomy validation" in md


def test_cli_validate_offline():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [sys.executable, "-m", "llmintent", "validate", "--format", "json"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=60,
    )
    # Exit 1 if any claim is killed (e.g. valid_instruments leak); still must parse.
    assert proc.stdout
    import json

    payload = json.loads(proc.stdout)
    assert "results" in payload
    assert "next_version" in payload
