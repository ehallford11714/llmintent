"""Vendored IntentIsolates core (offline, no external intentisolates required)."""

from __future__ import annotations

from llmintent.isolates._core.identify import identify_isolates
from llmintent.isolates._core.layers import assign_layers, soft_latentintent_layers, soft_llmintent_layers
from llmintent.isolates._core.motifs import form_motifs
from llmintent.isolates._core.report import build_report, report_to_json, report_to_markdown
from llmintent.isolates._core.span_burst import (
    CreativeBurstHopper,
    burst_path_from_text,
    identify_span_isolates,
    span_isolates_from_isolates,
    typology_path_entropy,
)
from llmintent.isolates._core.trajectory import trajectory_from_motifs
from llmintent.isolates._core.types import (
    ABSTRACT_LAYERS,
    BurstHop,
    BurstPath,
    Isolate,
    IsolateKind,
    IsolateReport,
    Motif,
    MotifTypology,
    ReasoningTrajectory,
    SpanIsolate,
    TextSpan,
    TrajectoryRole,
    TrajectoryStep,
    TypologyLabel,
)
from llmintent.isolates._core.typology import classify_typology
from llmintent.isolates._core.backends import available_backends

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "ABSTRACT_LAYERS",
    "BurstHop",
    "BurstPath",
    "CreativeBurstHopper",
    "Isolate",
    "IsolateKind",
    "IsolateReport",
    "LayerCausalResult",
    "LayerCausalSuite",
    "Motif",
    "MotifTypology",
    "ReasoningTrajectory",
    "SpanIsolate",
    "TextSpan",
    "TrajectoryRole",
    "TrajectoryStep",
    "TypologyLabel",
    "assign_layers",
    "available_backends",
    "build_report",
    "burst_path_from_text",
    "classify_typology",
    "form_motifs",
    "identify_isolates",
    "identify_span_isolates",
    "report_to_json",
    "report_to_markdown",
    "soft_latentintent_layers",
    "soft_llmintent_layers",
    "span_isolates_from_isolates",
    "trajectory_from_motifs",
    "typology_path_entropy",
]


def __getattr__(name: str):
    if name in ("LayerCausalSuite", "LayerCausalResult"):
        from llmintent.isolates._core.causal import LayerCausalResult, LayerCausalSuite

        return LayerCausalSuite if name == "LayerCausalSuite" else LayerCausalResult
    raise AttributeError(f"module 'llmintent.isolates._core' has no attribute {name!r}")
