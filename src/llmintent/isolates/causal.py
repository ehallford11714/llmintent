"""Re-export layer-causal / IV bridge (vendored or external intentisolates)."""

from __future__ import annotations

from typing import Any

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


def _backend():
    try:
        from intentisolates import causal as ext  # type: ignore

        return ext
    except ImportError:
        from llmintent.isolates._core import causal as vendored

        return vendored


def __getattr__(name: str) -> Any:
    mod = _backend()
    try:
        return getattr(mod, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'llmintent.isolates.causal' has no attribute {name!r}") from exc


_m = _backend()
CausalEdgeEstimate = _m.CausalEdgeEstimate
IndicationScore = _m.IndicationScore
LayerCausalResult = _m.LayerCausalResult
LayerCausalSuite = _m.LayerCausalSuite
MotifFeatureTable = _m.MotifFeatureTable
build_feature_frame = _m.build_feature_frame
column_name_for_isolate = _m.column_name_for_isolate
column_name_for_motif = _m.column_name_for_motif
estimate_indication = _m.estimate_indication
estimate_layer_iv = _m.estimate_layer_iv
