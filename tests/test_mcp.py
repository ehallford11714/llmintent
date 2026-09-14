"""Offline MCP dispatch — no model downloads, no 27B."""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path


def test_mcp_dispatch_offline(monkeypatch):
    monkeypatch.delenv("LLMINTENT_GUIDE_URL", raising=False)
    monkeypatch.delenv("LLMINTENT_GUIDE_KEY", raising=False)
    from llmintent.mcp import MCP_TOOLS, dispatch

    names = {t["name"] for t in MCP_TOOLS}
    assert {"li_status", "li_compile", "li_anatomy", "li_draft", "li_latent"} <= names

    status = dispatch("li_status")
    assert "vision" in status["regions"]
    assert status["version"]
    assert "li_anatomy" in status["tools"]

    compiled = dispatch("li_compile", {"text": "a dark shape rushing toward me"})
    assert "vision" in compiled["regions"]

    anatomy = dispatch(
        "li_anatomy",
        {"text": "I hear a song because a dark shape is looming."},
    )
    assert anatomy["trace"]["spans"]
    ids = {r["id"] for r in anatomy["regions"] if r.get("compiled") or r.get("occupancy")}
    assert "vision" in ids
    assert "auditory" in ids
    vision = next(r for r in anatomy["regions"] if r["id"] == "vision")
    assert vision["does"]
    assert vision["varies"]

    draft = dispatch(
        "li_draft",
        {"text": "I hear a song because a dark shape is looming."},
    )
    assert draft["backend"] == "template"
    lowered = draft["markdown"].lower()
    assert "vision" in lowered
    assert "auditory" in lowered

    unknown = dispatch("li_unknown")
    assert unknown.get("error")


def test_mcp_install_and_protocol():
    from llmintent.mcp.install import host_configs
    from llmintent.mcp.protocol import read_message

    hosts = host_configs()
    cursor = hosts["hosts"]["cursor"]["config"]["mcpServers"]["llmintent"]
    assert cursor["args"] == ["-m", "llmintent.mcp"]
    assert "PYTHONPATH" in cursor["env"]

    buf = StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
    msg = read_message(buf)
    assert msg is not None
    assert msg["method"] == "ping"


def test_cli_mcp_install_and_guide():
    root = Path(__file__).resolve().parents[1]
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONIOENCODING": "utf-8"}
    env["PYTHONPATH"] = str(root / "src") + (
        __import__("os").pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.pop("LLMINTENT_GUIDE_URL", None)
    env.pop("LLMINTENT_GUIDE_KEY", None)

    install = subprocess.run(
        [sys.executable, "-m", "llmintent", "mcp", "--install"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert install.returncode == 0, install.stderr
    payload = json.loads(install.stdout)
    assert "hosts" in payload

    guide = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmintent",
            "guide",
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
    assert guide.returncode == 0, guide.stderr
    draft = json.loads(guide.stdout)
    assert draft["backend"] == "template"
    assert "vision" in draft["markdown"].lower()
