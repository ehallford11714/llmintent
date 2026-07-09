"""Reasoning trajectories across layers from motifs and isolates."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from llmintent.isolates._core.layers import layer_name_for
from llmintent.isolates._core.motifs import _layer_sort_key, _typ
from llmintent.isolates._core.types import (
    ABSTRACT_LAYERS,
    Isolate,
    Motif,
    ReasoningTrajectory,
    TrajectoryRole,
    TrajectoryStep,
)

_CAVEATS = [
    "Motifs and trajectories are structural hypotheses, not validated cognitive mechanisms.",
    "Abstract L0–L4 layers are a reasoning scaffold, not necessarily model residual indices.",
    "Typed paths (e.g. goal->constraint->action) are template matches, not causal proofs.",
]


def trajectory_from_motifs(
    motifs: Sequence[Motif],
    isolates: Sequence[Isolate] | None = None,
    *,
    title: str = "Reasoning trajectory",
) -> ReasoningTrajectory:
    """
    Order motifs/isolates into a reasoning trajectory across layers
    (early lexical → mid latent → late goal/action).
    """
    isolates = list(isolates or [])
    by_id = {i.id: i for i in isolates}

    # Bucket isolates by layer
    layer_isos: dict[int, list[Isolate]] = defaultdict(list)
    for iso in isolates:
        layer_isos[_layer_sort_key(iso.layer)].append(iso)

    # Bucket motifs by primary (min) layer
    layer_motifs: dict[int, list[Motif]] = defaultdict(list)
    for m in motifs:
        if m.layers:
            key = min(_layer_sort_key(L) for L in m.layers)
        else:
            key = 2
        layer_motifs[key].append(m)

    all_layers = sorted(set(layer_isos) | set(layer_motifs) | set(ABSTRACT_LAYERS))
    # Keep only layers that have content, but always show path endpoints if any content
    content_layers = [L for L in all_layers if layer_isos.get(L) or layer_motifs.get(L)]
    if not content_layers:
        content_layers = [0, 2, 4]

    steps: list[TrajectoryStep] = []
    layer_path: list[int | str] = []
    motif_path: list[str] = []

    for idx, L in enumerate(content_layers):
        isos = layer_isos.get(L, [])
        mots = layer_motifs.get(L, [])
        role = _role_for_layer(L)
        iso_ids = [i.id for i in isos]
        mot_ids = [m.id for m in mots]
        for mid in mot_ids:
            if mid not in motif_path:
                motif_path.append(mid)
        layer_path.append(L)
        summary_bits = []
        if isos:
            labels = ", ".join(_typ(i) for i in isos[:4])
            summary_bits.append(f"isolates[{labels}]")
        if mots:
            pats = ", ".join((m.pattern or m.id) for m in mots[:3])
            summary_bits.append(f"motifs[{pats}]")
        steps.append(
            TrajectoryStep(
                index=idx,
                layer=L,
                layer_name=ABSTRACT_LAYERS.get(L, layer_name_for(L)),
                isolate_ids=iso_ids,
                motif_ids=mot_ids,
                role=role,
                summary="; ".join(summary_bits) or "(empty)",
            )
        )

    ascii_diagram = _ascii(steps, by_id)
    mermaid = _mermaid(steps, motifs, by_id)
    summary_md = _summary_markdown(title, steps, motifs)

    return ReasoningTrajectory(
        steps=steps,
        layer_path=layer_path,
        motif_path=motif_path,
        summary_markdown=summary_md,
        mermaid=mermaid,
        ascii_diagram=ascii_diagram,
        caveats=list(_CAVEATS),
        metadata={
            "n_steps": len(steps),
            "n_motifs": len(motifs),
            "n_isolates": len(isolates),
            "layer_role_note": (
                "Early layers surface lexical/affective isolates; mid layers "
                "host latent/confounder structure; late layers concentrate "
                "goal, constraint, action, and outcome — aiding interpretation "
                "of each layer's role along the reasoning path."
            ),
        },
    )


def _role_for_layer(L: int) -> TrajectoryRole:
    if L <= 1:
        return TrajectoryRole.EARLY_LEXICAL
    if L <= 2:
        return TrajectoryRole.MID_LATENT
    return TrajectoryRole.LATE_GOAL


def _ascii(steps: Sequence[TrajectoryStep], by_id: dict[str, Isolate]) -> str:
    lines = ["Reasoning trajectory (ASCII)", ""]
    for i, step in enumerate(steps):
        branch = "+-" if i == len(steps) - 1 else "|-"
        name = step.layer_name or f"L{step.layer}"
        role = step.role.value if hasattr(step.role, "value") else str(step.role)
        lines.append(f"{branch} [{name}] ({role})")
        for iid in step.isolate_ids[:5]:
            iso = by_id.get(iid)
            label = iso.label[:48] if iso else iid
            typ = _typ(iso) if iso else "?"
            lines.append(f"   * {typ}: {label}")
        for mid in step.motif_ids[:3]:
            lines.append(f"   * motif {mid}")
        if i < len(steps) - 1:
            lines.append("|")
    return "\n".join(lines)


def _mermaid(
    steps: Sequence[TrajectoryStep],
    motifs: Sequence[Motif],
    by_id: dict[str, Isolate],
) -> str:
    lines = ["flowchart TD"]
    for step in steps:
        nid = f"L{step.layer}"
        name = (step.layer_name or nid).replace('"', "'")
        lines.append(f'  {nid}["{name}"]')
    for a, b in zip(steps, steps[1:]):
        lines.append(f"  L{a.layer} --> L{b.layer}")
    # Annotate typed motifs as dashed links between first/last member layers
    for m in motifs:
        if not m.member_ids or len(m.member_ids) < 2:
            continue
        first = by_id.get(m.member_ids[0])
        last = by_id.get(m.member_ids[-1])
        if not first or not last:
            continue
        la, lb = _layer_sort_key(first.layer), _layer_sort_key(last.layer)
        if la != lb:
            pat = (m.pattern or m.id).replace('"', "'")
            lines.append(f"  L{la} -.->|{pat}| L{lb}")
    return "\n".join(lines)


def _summary_markdown(
    title: str,
    steps: Sequence[TrajectoryStep],
    motifs: Sequence[Motif],
) -> str:
    lines = [
        f"### {title}",
        "",
        "Trajectory moves from early lexical/affective isolates through mid "
        "latent structure toward late goal/constraint/action layers. Motifs "
        "that bridge layers highlight how local isolates compose into a "
        "longer reasoning path — useful for interpreting **layer roles** "
        "(what each band contributes) rather than treating layers as opaque indices.",
        "",
        "**Steps:**",
    ]
    for step in steps:
        role = step.role.value if hasattr(step.role, "value") else str(step.role)
        lines.append(
            f"1. **{step.layer_name}** (`{role}`): {step.summary}"
        )
    if motifs:
        lines.append("")
        lines.append(f"**Motifs involved:** {len(motifs)} "
                      f"(typed={sum(1 for m in motifs if str(getattr(m.typology, 'value', m.typology)) == 'typed_path')}).")
    lines.append("")
    lines.append("> Caveat: structural hypothesis only — not a claim of true model cognition.")
    return "\n".join(lines)
