"""Optional soft re-export of intentisolates (PyPI: intentisolates).

Install both packages, then::

    from llmintent.isolates import identify_isolates, form_motifs, trajectory_from_motifs

If ``intentisolates`` is not installed, imports raise a clear error.
"""

from __future__ import annotations

try:
    from intentisolates import (  # noqa: F401
        ABSTRACT_LAYERS,
        Isolate,
        IsolateKind,
        IsolateReport,
        Motif,
        MotifTypology,
        ReasoningTrajectory,
        TrajectoryRole,
        TrajectoryStep,
        TypologyLabel,
        __version__ as intentisolates_version,
        assign_layers,
        available_backends,
        build_report,
        classify_typology,
        form_motifs,
        identify_isolates,
        report_to_json,
        report_to_markdown,
        trajectory_from_motifs,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "llmintent.isolates requires the intentisolates package. "
        "Install with: pip install intentisolates"
    ) from exc

__all__ = [
    "ABSTRACT_LAYERS",
    "Isolate",
    "IsolateKind",
    "IsolateReport",
    "Motif",
    "MotifTypology",
    "ReasoningTrajectory",
    "TrajectoryRole",
    "TrajectoryStep",
    "TypologyLabel",
    "assign_layers",
    "available_backends",
    "build_report",
    "classify_typology",
    "form_motifs",
    "identify_isolates",
    "intentisolates_version",
    "report_to_json",
    "report_to_markdown",
    "trajectory_from_motifs",
]
