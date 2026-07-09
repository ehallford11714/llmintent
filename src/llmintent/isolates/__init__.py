"""llmintent.isolates — suite home for intent isolates + typology.

Prefers an installed ``intentisolates`` package when present; otherwise uses the
vendored offline implementation under ``llmintent.isolates._core`` so
``pip install llmintent`` works standalone.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.1.0"

_SOURCE = "vendored"
_BACKEND: Any = None


def _load_backend():
    global _BACKEND, _SOURCE
    if _BACKEND is not None:
        return _BACKEND
    try:
        import intentisolates as ext  # type: ignore

        _BACKEND = ext
        _SOURCE = "intentisolates"
        return _BACKEND
    except ImportError:
        from llmintent.isolates import _core as vendored

        _BACKEND = vendored
        _SOURCE = "vendored"
        return _BACKEND


def __getattr__(name: str) -> Any:
    if name == "backend_source":
        _load_backend()
        return _SOURCE
    if name in ("LayerCausalSuite", "LayerCausalResult"):
        from llmintent.isolates.causal import LayerCausalResult, LayerCausalSuite

        return LayerCausalSuite if name == "LayerCausalSuite" else LayerCausalResult
    backend = _load_backend()
    try:
        return getattr(backend, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'llmintent.isolates' has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"backend_source"})


# Eagerly bind common symbols for static analyzers / star-import friendliness.
_b = _load_backend()

ABSTRACT_LAYERS = _b.ABSTRACT_LAYERS
Isolate = _b.Isolate
IsolateKind = _b.IsolateKind
IsolateReport = _b.IsolateReport
Motif = _b.Motif
MotifTypology = _b.MotifTypology
ReasoningTrajectory = _b.ReasoningTrajectory
TrajectoryRole = _b.TrajectoryRole
TrajectoryStep = _b.TrajectoryStep
TypologyLabel = _b.TypologyLabel
assign_layers = _b.assign_layers
available_backends = _b.available_backends
build_report = _b.build_report
classify_typology = _b.classify_typology
form_motifs = _b.form_motifs
identify_isolates = _b.identify_isolates
report_to_json = _b.report_to_json
report_to_markdown = _b.report_to_markdown
soft_latentintent_layers = _b.soft_latentintent_layers
soft_llmintent_layers = _b.soft_llmintent_layers
trajectory_from_motifs = _b.trajectory_from_motifs

__all__ = [
    "__version__",
    "ABSTRACT_LAYERS",
    "Isolate",
    "IsolateKind",
    "IsolateReport",
    "LayerCausalResult",
    "LayerCausalSuite",
    "Motif",
    "MotifTypology",
    "ReasoningTrajectory",
    "TrajectoryRole",
    "TrajectoryStep",
    "TypologyLabel",
    "assign_layers",
    "available_backends",
    "backend_source",
    "build_report",
    "classify_typology",
    "form_motifs",
    "identify_isolates",
    "report_to_json",
    "report_to_markdown",
    "soft_latentintent_layers",
    "soft_llmintent_layers",
    "trajectory_from_motifs",
]
