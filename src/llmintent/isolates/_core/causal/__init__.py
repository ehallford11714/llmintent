"""Layer-motif ↔ AutoCausal / CausalIV bridge.

Convert isolates and motifs into tabular features, estimate layer→output
indication (association) vs causation (IV / 2SLS), and emit structured reports.
"""

from __future__ import annotations

from llmintent.isolates._core.causal.features import (
    MotifFeatureTable,
    build_feature_frame,
    column_name_for_isolate,
    column_name_for_motif,
)
from llmintent.isolates._core.causal.iv_layers import (
    CausalEdgeEstimate,
    IndicationScore,
    estimate_indication,
    estimate_layer_iv,
)
from llmintent.isolates._core.causal.suite import LayerCausalResult, LayerCausalSuite

__all__ = [
    "CausalEdgeEstimate",
    "IndicationScore",
    "LayerCausalResult",
    "LayerCausalSuite",
    "MotifFeatureTable",
    "build_feature_frame",
    "column_name_for_isolate",
    "column_name_for_motif",
    "estimate_indication",
    "estimate_layer_iv",
]
