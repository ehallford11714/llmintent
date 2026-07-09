"""Form motifs from layered isolates (co-occurrence, sequences, typed paths)."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable, Sequence

from llmintent.isolates._core.types import (
    TYPED_MOTIF_TEMPLATES,
    Isolate,
    Motif,
    MotifTypology,
    TrajectoryRole,
)


def form_motifs(
    isolates: Sequence[Isolate],
    layers: Sequence[int | str] | None = None,
    *,
    min_support: float = 0.15,
    include_typed: bool = True,
    include_graph_motifs: bool = True,
) -> list[Motif]:
    """
    Detect motifs: co-occurrence, sequential patterns, typed paths, and
    simple graph motifs (chains / triangles over isolate ids).

    Epistemic note: motifs are **structural hypotheses**, not proven
    cognitive mechanisms.
    """
    if not isolates:
        return []

    by_id = {iso.id: iso for iso in isolates}
    filtered = list(isolates)
    if layers is not None:
        layer_set = {_norm_layer(x) for x in layers}
        filtered = [i for i in isolates if _norm_layer(i.layer) in layer_set]
        if not filtered:
            filtered = list(isolates)

    motifs: list[Motif] = []
    motifs.extend(_cooccurrence_motifs(filtered, min_support=min_support))
    motifs.extend(_sequence_motifs(filtered))
    if include_typed:
        motifs.extend(_typed_path_motifs(filtered))
    if include_graph_motifs:
        motifs.extend(_graph_structure_motifs(filtered))
    motifs.extend(_layer_bridge_motifs(filtered))

    # Assign trajectory roles from member layers
    for m in motifs:
        m.trajectory_role = _role_for_layers(m.layers)

    # Deduplicate by pattern + member frozenset
    return _dedupe(motifs)


def _cooccurrence_motifs(
    isolates: Sequence[Isolate],
    *,
    min_support: float,
) -> list[Motif]:
    """Pairs sharing a layer (or adjacent layers) as co-occurrence motifs."""
    by_layer: dict[int | str, list[Isolate]] = defaultdict(list)
    for iso in isolates:
        by_layer[_norm_layer(iso.layer if iso.layer is not None else "NA")].append(iso)

    motifs: list[Motif] = []
    mid = 0
    n = max(len(isolates), 1)
    for layer, members in by_layer.items():
        if len(members) < 2:
            continue
        for a, b in combinations(members, 2):
            support = 2.0 / n
            if support < min_support and n > 4:
                continue
            ta = _typ(a)
            tb = _typ(b)
            motifs.append(
                Motif(
                    id=f"co_{mid}",
                    typology=MotifTypology.CO_OCCURRENCE,
                    member_ids=[a.id, b.id],
                    layers=[layer],
                    support=round(support, 3),
                    confidence=round(min(0.9, 0.4 + 0.2 * support * n), 3),
                    pattern=f"{ta}+{tb}@{layer}",
                    rationale=f"Co-occur in layer {layer}: {a.label[:40]} | {b.label[:40]}",
                )
            )
            mid += 1
    return motifs


def _sequence_motifs(isolates: Sequence[Isolate]) -> list[Motif]:
    """Order isolates by span start / id and emit adjacent pairs as sequences."""
    ordered = sorted(
        isolates,
        key=lambda i: (
            i.span[0] if i.span else 10**9,
            _layer_sort_key(i.layer),
            i.id,
        ),
    )
    motifs: list[Motif] = []
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        layers = _uniq_layers([a.layer, b.layer])
        motifs.append(
            Motif(
                id=f"seq_{i}",
                typology=MotifTypology.SEQUENCE,
                member_ids=[a.id, b.id],
                layers=layers,
                support=round(2.0 / max(len(isolates), 1), 3),
                confidence=0.55,
                pattern=f"{_typ(a)}->{_typ(b)}",
                rationale=f"Sequential adjacency: '{a.label[:32]}' -> '{b.label[:32]}'",
            )
        )
    return motifs


def _typed_path_motifs(isolates: Sequence[Isolate]) -> list[Motif]:
    """Match known typology templates e.g. goal->constraint->action."""
    by_typ: dict[str, list[Isolate]] = defaultdict(list)
    for iso in isolates:
        by_typ[_typ(iso)].append(iso)

    motifs: list[Motif] = []
    mid = 0
    for template in TYPED_MOTIF_TEMPLATES:
        pools = [by_typ.get(t, []) for t in template]
        if any(len(p) == 0 for p in pools):
            continue
        # Greedy: take first of each type, prefer increasing layer order
        chosen: list[Isolate] = []
        ok = True
        last_layer = -1
        for pool in pools:
            pool_sorted = sorted(pool, key=lambda x: _layer_sort_key(x.layer))
            pick = None
            for cand in pool_sorted:
                if cand in chosen:
                    continue
                lk = _layer_sort_key(cand.layer)
                if lk >= last_layer or last_layer < 0:
                    pick = cand
                    last_layer = lk
                    break
            if pick is None:
                pick = pool_sorted[0]
            if pick in chosen:
                ok = False
                break
            chosen.append(pick)
        if not ok or len(chosen) != len(template):
            continue
        layers = _uniq_layers([c.layer for c in chosen])
        pattern = "->".join(template)
        motifs.append(
            Motif(
                id=f"typed_{mid}",
                typology=MotifTypology.TYPED_PATH,
                member_ids=[c.id for c in chosen],
                layers=layers,
                support=round(len(chosen) / max(len(isolates), 1), 3),
                confidence=0.72,
                pattern=pattern,
                rationale=f"Typed motif template matched: {pattern}",
                metadata={"template": list(template)},
            )
        )
        mid += 1
    return motifs


def _graph_structure_motifs(isolates: Sequence[Isolate]) -> list[Motif]:
    """
    Build a soft graph: connect isolates in adjacent layers or sequential spans,
    then find chains (length 3) and triangles.
    """
    if len(isolates) < 3:
        return []

    ids = [i.id for i in isolates]
    id_to = {i.id: i for i in isolates}
    # Edges: sequential + same/adjacent layer
    edges: set[frozenset[str]] = set()
    ordered = sorted(isolates, key=lambda i: (i.span[0] if i.span else 0, i.id))
    for a, b in zip(ordered, ordered[1:]):
        edges.add(frozenset((a.id, b.id)))
    for a, b in combinations(isolates, 2):
        la, lb = _layer_sort_key(a.layer), _layer_sort_key(b.layer)
        if abs(la - lb) <= 1:
            edges.add(frozenset((a.id, b.id)))

    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)

    motifs: list[Motif] = []
    mid = 0
    # Triangles
    for a, b, c in combinations(ids, 3):
        if b in adj[a] and c in adj[b] and c in adj[a]:
            members = [a, b, c]
            layers = _uniq_layers([id_to[x].layer for x in members])
            motifs.append(
                Motif(
                    id=f"tri_{mid}",
                    typology=MotifTypology.TRIANGLE,
                    member_ids=members,
                    layers=layers,
                    support=0.3,
                    confidence=0.5,
                    pattern="triangle",
                    rationale="Triangle motif over adjacent/co-layer isolates",
                )
            )
            mid += 1
            if mid >= 5:
                break

    # Chains of length 3 (path a-b-c without requiring a-c)
    chain_i = 0
    for b in ids:
        nbrs = sorted(adj[b])
        for a, c in combinations(nbrs, 2):
            if frozenset((a, c)) in edges:
                continue  # that's a triangle, already counted
            members = [a, b, c]
            # Prefer increasing layer
            members_sorted = sorted(members, key=lambda x: _layer_sort_key(id_to[x].layer))
            layers = _uniq_layers([id_to[x].layer for x in members_sorted])
            motifs.append(
                Motif(
                    id=f"chain_{chain_i}",
                    typology=MotifTypology.CHAIN,
                    member_ids=members_sorted,
                    layers=layers,
                    support=0.25,
                    confidence=0.48,
                    pattern=(
                        f"{_typ(id_to[members_sorted[0]])}->"
                        f"{_typ(id_to[members_sorted[1]])}->"
                        f"{_typ(id_to[members_sorted[2]])}"
                    ),
                    rationale="Chain motif across adjacent layers/spans",
                )
            )
            chain_i += 1
            if chain_i >= 8:
                break
        if chain_i >= 8:
            break
    return motifs


def _layer_bridge_motifs(isolates: Sequence[Isolate]) -> list[Motif]:
    """Pairs that connect non-adjacent layers (bridge role)."""
    motifs: list[Motif] = []
    mid = 0
    for a, b in combinations(isolates, 2):
        la, lb = _layer_sort_key(a.layer), _layer_sort_key(b.layer)
        if abs(la - lb) >= 2:
            motifs.append(
                Motif(
                    id=f"bridge_{mid}",
                    typology=MotifTypology.LAYER_BRIDGE,
                    member_ids=[a.id, b.id],
                    layers=_uniq_layers([a.layer, b.layer]),
                    support=0.2,
                    confidence=0.45,
                    pattern=f"bridge:{_typ(a)}<->{_typ(b)}",
                    rationale=f"Cross-layer bridge L{la}<->L{lb}",
                    trajectory_role=TrajectoryRole.BRIDGE,
                )
            )
            mid += 1
            if mid >= 6:
                break
    return motifs


def _role_for_layers(layers: Sequence[int | str]) -> TrajectoryRole:
    if not layers:
        return TrajectoryRole.UNKNOWN
    keys = [_layer_sort_key(x) for x in layers]
    avg = sum(keys) / len(keys)
    span = max(keys) - min(keys)
    if span >= 2:
        return TrajectoryRole.BRIDGE
    if avg <= 1.0:
        return TrajectoryRole.EARLY_LEXICAL
    if avg <= 2.5:
        return TrajectoryRole.MID_LATENT
    return TrajectoryRole.LATE_GOAL


def _dedupe(motifs: Sequence[Motif]) -> list[Motif]:
    seen: set[tuple] = set()
    out: list[Motif] = []
    for m in motifs:
        key = (_typ_m(m), frozenset(m.member_ids), m.pattern)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _typ(iso: Isolate) -> str:
    return iso.typology.value if hasattr(iso.typology, "value") else str(iso.typology)


def _typ_m(m: Motif) -> str:
    return m.typology.value if hasattr(m.typology, "value") else str(m.typology)


def _norm_layer(layer: int | str | None) -> int | str:
    if layer is None:
        return "NA"
    return layer


def _layer_sort_key(layer: int | str | None) -> int:
    if layer is None:
        return 99
    if isinstance(layer, int):
        return layer
    s = str(layer)
    if s.isdigit():
        return int(s)
    import re
    m = re.match(r"L(\d+)", s, re.I)
    if m:
        return int(m.group(1))
    return 50


def _uniq_layers(layers: Iterable[int | str | None]) -> list[int | str]:
    out: list[int | str] = []
    seen: set = set()
    for L in layers:
        if L is None:
            continue
        if L in seen:
            continue
        seen.add(L)
        out.append(L)
    return out
