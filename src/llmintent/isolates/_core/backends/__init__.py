"""Optional backends (rule always available; HF / llmintent soft)."""

from __future__ import annotations

from typing import Any


def available_backends() -> dict[str, bool]:
    out = {"rule": True, "hf": False, "llmintent": False, "latentintent": False}
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        out["hf"] = True
    except Exception:
        pass
    try:
        import llmintent  # noqa: F401

        out["llmintent"] = True
    except Exception:
        pass
    try:
        import latentintent  # noqa: F401

        out["latentintent"] = True
    except Exception:
        try:
            import latentintentinspect  # noqa: F401

            out["latentintent"] = True
        except Exception:
            pass
    return out


def describe_backend(name: str = "rule") -> dict[str, Any]:
    avail = available_backends()
    return {
        "name": name,
        "available": bool(avail.get(name, False)),
        "all": avail,
        "note": "rule backend is always online; others are optional soft imports",
    }
