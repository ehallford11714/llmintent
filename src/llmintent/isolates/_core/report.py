"""Report builders (JSON / markdown)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from llmintent.isolates._core.identify import identify_isolates
from llmintent.isolates._core.motifs import form_motifs
from llmintent.isolates._core.trajectory import trajectory_from_motifs
from llmintent.isolates._core.types import Isolate, IsolateReport, Motif, ReasoningTrajectory

DEFAULT_CAVEATS = [
    "Rule/heuristic backend — offline structural analysis, not ground-truth intent.",
    "Motifs are compositional hypotheses; support/confidence are heuristic scores.",
    "Layer indices may be abstract (L0–L4) unless bound to a model residual stream.",
]


def build_report(
    *,
    text: str | None = None,
    features: Any = None,
    graph: Any = None,
    isolates: Sequence[Isolate] | None = None,
    include_motifs: bool = True,
    include_trajectory: bool = True,
    backend: str = "rule",
    **kwargs: Any,
) -> IsolateReport:
    """Build a full IsolateReport from inputs or precomputed isolates."""
    if isolates is None:
        isolates = identify_isolates(
            text=text,
            features=features,
            graph=graph,
            backend=backend,
            **kwargs,
        )
    else:
        isolates = list(isolates)

    motifs: list[Motif] = []
    traj: ReasoningTrajectory | None = None
    if include_motifs or include_trajectory:
        motifs = form_motifs(isolates)
    if include_trajectory:
        traj = trajectory_from_motifs(motifs, isolates)

    summary = ""
    if text:
        summary = text if len(text) <= 120 else text[:117] + "..."
    elif features is not None:
        summary = f"features({len(features) if hasattr(features, '__len__') else '?'})"
    elif graph is not None:
        summary = "graph(...)"

    caveats = list(DEFAULT_CAVEATS)
    if traj:
        caveats.extend(traj.caveats)

    return IsolateReport(
        isolates=list(isolates),
        motifs=motifs if include_motifs else [],
        trajectory=traj if include_trajectory else None,
        backend=backend,
        input_summary=summary,
        caveats=_uniq(caveats),
        metadata={"include_motifs": include_motifs, "include_trajectory": include_trajectory},
    )


def report_to_json(report: IsolateReport, path: str | Path | None = None, *, indent: int = 2) -> str:
    payload = json.dumps(report.to_dict(), indent=indent, ensure_ascii=False)
    if path is not None:
        Path(path).write_text(payload, encoding="utf-8")
    return payload


def report_to_markdown(report: IsolateReport, path: str | Path | None = None) -> str:
    md = report.to_markdown()
    if path is not None:
        Path(path).write_text(md, encoding="utf-8")
    return md


def _uniq(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
