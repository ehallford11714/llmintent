"""MCP host install snippets for LLMIntent."""

from __future__ import annotations

import sys


def python_exe() -> str:
    return sys.executable or "python"


def server_block() -> dict:
    return {
        "command": python_exe(),
        "args": ["-m", "llmintent.mcp"],
        "env": {
            "PYTHONUTF8": "1",
            "PYTHONPATH": "${workspaceFolder}/src",
        },
    }


def host_configs() -> dict:
    block = server_block()
    return {
        "module": "python -m llmintent.mcp",
        "script": "llmintent-mcp",
        "hosts": {
            "cursor": {
                "path": ".cursor/mcp.json",
                "config": {"mcpServers": {"llmintent": block}},
            },
            "vscode": {
                "path": ".vscode/mcp.json",
                "config": {"servers": {"llmintent": {"type": "stdio", **block}}},
            },
            "claude_code": {
                "path": ".mcp.json",
                "config": {"mcpServers": {"llmintent": block}},
            },
            "generic": {
                "path": "mcp.json",
                "config": {"mcpServers": {"llmintent": block}},
            },
        },
    }
