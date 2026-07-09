"""llmintent.iv_motifs — IV / AutoCausal layer-causal bridge for motifs.

Indication (association) vs causation (IV / 2SLS). Soft-depends on
``causaliv`` / ``autocausal`` when available; stdlib Wald IV otherwise.
"""

from __future__ import annotations

from llmintent.isolates.causal import (
    CausalEdgeEstimate,
    IndicationScore,
    LayerCausalResult,
    LayerCausalSuite,
    MotifFeatureTable,
    build_feature_frame,
    estimate_indication,
    estimate_layer_iv,
)

# Alias used in docs / CLI naming
CausalLayers = LayerCausalSuite

__all__ = [
    "CausalEdgeEstimate",
    "CausalLayers",
    "IndicationScore",
    "LayerCausalResult",
    "LayerCausalSuite",
    "MotifFeatureTable",
    "build_feature_frame",
    "estimate_indication",
    "estimate_layer_iv",
]
