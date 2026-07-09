"""llmintent.latent — soft hooks for LatentIntentInspect / latentintent.

LatentIntentInspect is not required. When installed, this module surfaces
availability metadata and any stable public symbols. Otherwise it provides
a lightweight stub so suite imports never fail.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "available",
    "backend_name",
    "describe",
    "soft_latentintent_layers",
]

_AVAILABLE = False
_BACKEND_NAME: str | None = None
_MOD: Any = None

for _name in ("latentintent", "latentintentinspect"):
    try:
        _MOD = __import__(_name)
        _AVAILABLE = True
        _BACKEND_NAME = _name
        break
    except ImportError:
        continue


def available() -> bool:
    return _AVAILABLE


def backend_name() -> str | None:
    return _BACKEND_NAME


def describe() -> dict[str, Any]:
    """Return suite status for latent thought inspection."""
    info: dict[str, Any] = {
        "available": _AVAILABLE,
        "backend": _BACKEND_NAME,
        "note": (
            "LatentIntentInspect is an optional extractable companion. "
            "Install when published; until then llmintent.latent is a soft stub."
        ),
    }
    if _MOD is not None:
        info["version"] = getattr(_MOD, "__version__", None)
        info["exports"] = [n for n in dir(_MOD) if not n.startswith("_")][:40]
    return info


def soft_latentintent_layers(**kwargs: Any) -> dict[str, Any] | None:
    """Delegate to isolates layer soft-hook (never raises)."""
    from llmintent.isolates import soft_latentintent_layers as _hook

    return _hook(**kwargs)


def __getattr__(name: str) -> Any:
    if _MOD is not None and hasattr(_MOD, name):
        return getattr(_MOD, name)
    raise AttributeError(
        f"llmintent.latent has no attribute {name!r} "
        f"(latent backend available={_AVAILABLE!r}). "
        "Install LatentIntentInspect / latentintent when published."
    )
