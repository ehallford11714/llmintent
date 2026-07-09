"""llmintent.latent — latent thought inspection for the suite.

Resolution order
----------------
1. Prefer installed extractable ``latentintent`` (LatentIntentInspect) when present.
2. Else use vendored offline API under ``llmintent.latent_vendor``.

Epistemic note: inspects **activation correlates / probes / heuristics** —
not human thoughts, and not proven ground-truth model "mind reading."
Every ThoughtReport includes caveats.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EPISTEMIC_CAVEATS",
    "IntentHypothesis",
    "LayerSaliency",
    "ThoughtReport",
    "available",
    "backend_name",
    "build_thought_report",
    "describe",
    "inspect_text",
    "soft_latentintent_layers",
]

_EXTERNAL = False
_BACKEND_NAME: str | None = None
_MOD: Any = None

for _name in ("latentintent", "latentintentinspect"):
    try:
        _MOD = __import__(_name)
        _EXTERNAL = True
        _BACKEND_NAME = _name
        break
    except ImportError:
        continue

if _MOD is None:
    from llmintent import latent_vendor as _MOD

    _BACKEND_NAME = "llmintent.latent_vendor"
    _EXTERNAL = False

# Re-export stable API from resolved backend
EPISTEMIC_CAVEATS = getattr(_MOD, "EPISTEMIC_CAVEATS", ())
IntentHypothesis = _MOD.IntentHypothesis
LayerSaliency = _MOD.LayerSaliency
ThoughtReport = _MOD.ThoughtReport
build_thought_report = _MOD.build_thought_report
inspect_text = _MOD.inspect_text


def available() -> bool:
    """True when a latent inspect backend is loaded (always True since 1.2.0)."""
    return _MOD is not None


def backend_name() -> str | None:
    return _BACKEND_NAME


def describe() -> dict[str, Any]:
    """Return suite status for latent thought inspection."""
    info: dict[str, Any] = {
        "available": True,
        "backend": _BACKEND_NAME,
        "external": _EXTERNAL,
        "note": (
            "Latent thought inspection reports hypothesized correlates "
            "(rules / probes / SAE-lite stubs), not proven model minds. "
            "Install extractable `latentintent` for HF residual capture; "
            "vendored offline path always ships in llmintent."
        ),
    }
    info["version"] = getattr(_MOD, "__version__", None)
    info["exports"] = [n for n in dir(_MOD) if not n.startswith("_")][:40]
    return info


def soft_latentintent_layers(**kwargs: Any) -> dict[str, Any] | None:
    """Delegate to isolates layer soft-hook (never raises)."""
    from llmintent.isolates import soft_latentintent_layers as _hook

    return _hook(**kwargs)


def __getattr__(name: str) -> Any:
    if hasattr(_MOD, name):
        return getattr(_MOD, name)
    raise AttributeError(f"module 'llmintent.latent' has no attribute {name!r}")
