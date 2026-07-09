"""Vendored offline latent-thought inspection (no external latentintent required).

Epistemic note: reports hypothesized correlates / heuristics — not mind-reading.
Prefer installed ``latentintent`` when present; this package is the fallback.
"""

from __future__ import annotations

from llmintent.latent_vendor.report import ThoughtReport, build_thought_report, inspect_text
from llmintent.latent_vendor.types import EPISTEMIC_CAVEATS, IntentHypothesis, LayerSaliency

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "EPISTEMIC_CAVEATS",
    "IntentHypothesis",
    "LayerSaliency",
    "ThoughtReport",
    "build_thought_report",
    "inspect_text",
]
