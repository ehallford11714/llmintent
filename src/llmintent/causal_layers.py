"""llmintent.causal_layers — alias of ``llmintent.iv_motifs``."""

from __future__ import annotations

from llmintent.iv_motifs import (
    CausalEdgeEstimate,
    CausalLayers,
    IndicationScore,
    LayerCausalResult,
    LayerCausalSuite,
    MotifFeatureTable,
    build_feature_frame,
    estimate_indication,
    estimate_layer_iv,
)

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
