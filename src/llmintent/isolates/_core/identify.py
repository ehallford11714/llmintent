"""Identify isolates from text, feature vectors, or small graphs."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

from llmintent.isolates._core.layers import assign_layers
from llmintent.isolates._core.types import Isolate, IsolateKind
from llmintent.isolates._core.typology import classify_typology

# Split on clause-ish boundaries while keeping short intent phrases.
_CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?;])\s+|\n+|\s*(?:,\s+and\s+|;\s+|—\s*|–\s*)\s*",
    re.I,
)
_INTENTISH = re.compile(
    r"\b(i want|i need|please|must|should|cannot|can't|so that|"
    r"in order to|because|using|via|feel|goal|aim)\b",
    re.I,
)


def identify_isolates(
    text: str | None = None,
    features: Sequence[float] | Mapping[str, float] | None = None,
    graph: Mapping[str, Any] | Sequence[tuple[str, str]] | None = None,
    *,
    backend: str = "rule",
    assign_layer: bool = True,
    layer_strategy: str = "abstract",
    min_feature_z: float = 2.0,
    **kwargs: Any,
) -> list[Isolate]:
    """
    Identify isolates from one or more modalities.

    Parameters
    ----------
    text :
        Free text → lexical/semantic phrase isolates.
    features :
        Activation / KPI vector (list or name→value map) → feature isolates.
    graph :
        Edge list or ``{"nodes": [...], "edges": [[u,v], ...]}`` → causal/orphan isolates.
    backend :
        ``rule`` (default, offline) or ``hf`` / ``llmintent`` (soft, optional).
    """
    isolates: list[Isolate] = []
    if text is not None and str(text).strip():
        isolates.extend(_from_text(str(text), backend=backend, **kwargs))
    if features is not None:
        isolates.extend(_from_features(features, min_z=min_feature_z, **kwargs))
    if graph is not None:
        isolates.extend(_from_graph(graph, **kwargs))

    # Classify typology
    classified = [classify_typology(iso) for iso in isolates]

    if assign_layer:
        classified = assign_layers(classified, strategy=layer_strategy)

    # Optional soft enrichment (does not replace rule results)
    if backend in ("hf", "llmintent", "soft"):
        classified = _soft_enrich(classified, text=text, backend=backend)

    return classified


def _from_text(text: str, *, backend: str = "rule", **kwargs: Any) -> list[Isolate]:
    text = text.strip()
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(text) if p and p.strip()]
    if len(parts) <= 1:
        # Also split on " and " for short multi-intent lines
        alt = [p.strip() for p in re.split(r"\s+and\s+", text) if p.strip()]
        if len(alt) > 1:
            parts = alt

    # Keep intent-ish clauses; if none match, keep all non-trivial parts
    scored = []
    for p in parts:
        if len(p) < 2:
            continue
        boost = 1.0 if _INTENTISH.search(p) else 0.0
        scored.append((boost, p))
    if any(b > 0 for b, _ in scored):
        # Prefer intent-ish but keep others as weaker isolates
        candidates = [p for _, p in scored]
    else:
        candidates = [p for _, p in scored] or [text]

    isolates: list[Isolate] = []
    for i, phrase in enumerate(candidates):
        start = text.find(phrase)
        end = start + len(phrase) if start >= 0 else None
        span = (start, end) if start >= 0 and end is not None else None
        isolates.append(
            Isolate(
                id=f"text_{i}",
                kind=IsolateKind.TEXT,
                label=phrase,
                span=span,
                source="rule",
                metadata={"text_len": len(text), "backend": backend},
            )
        )
    return isolates


def _from_features(
    features: Sequence[float] | Mapping[str, float],
    *,
    min_z: float = 2.0,
    top_k: int = 8,
    **kwargs: Any,
) -> list[Isolate]:
    if isinstance(features, Mapping):
        items = [(str(k), float(v)) for k, v in features.items()]
    else:
        items = [(f"f{i}", float(v)) for i, v in enumerate(features)]

    if not items:
        return []

    vals = [v for _, v in items]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
    std = math.sqrt(var) if var > 0 else 1.0

    ranked: list[tuple[float, str, float, float]] = []
    for name, v in items:
        z = (v - mean) / std
        # sparsity proxy: how extreme vs mean
        sparsity = min(1.0, abs(z) / 4.0)
        ranked.append((abs(z), name, v, sparsity))
    ranked.sort(reverse=True)

    isolates: list[Isolate] = []
    for i, (az, name, v, sparsity) in enumerate(ranked[:top_k]):
        if az < min_z and i > 0:
            # Always keep the top-1 even if below threshold
            continue
        isolates.append(
            Isolate(
                id=f"feat_{i}_{name}",
                kind=IsolateKind.FEATURE,
                label=f"{name}={v:.4g}",
                source="rule",
                metadata={
                    "feature": name,
                    "value": v,
                    "zscore": (v - mean) / std,
                    "sparsity": sparsity,
                    "mean": mean,
                    "std": std,
                },
            )
        )
    return isolates


def _from_graph(
    graph: Mapping[str, Any] | Sequence[tuple[str, str]],
    **kwargs: Any,
) -> list[Isolate]:
    nodes: set[str] = set()
    edges: list[tuple[str, str]] = []

    if isinstance(graph, Mapping):
        raw_nodes = graph.get("nodes") or []
        raw_edges = graph.get("edges") or []
        for n in raw_nodes:
            nodes.add(str(n))
        for e in raw_edges:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                u, v = str(e[0]), str(e[1])
                edges.append((u, v))
                nodes.add(u)
                nodes.add(v)
            elif isinstance(e, Mapping):
                u, v = str(e.get("source", e.get("u"))), str(e.get("target", e.get("v")))
                edges.append((u, v))
                nodes.add(u)
                nodes.add(v)
    else:
        for e in graph:
            u, v = str(e[0]), str(e[1])
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)

    degree: dict[str, int] = {n: 0 for n in nodes}
    for u, v in edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    # Connected components (undirected)
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    seen: set[str] = set()
    components: list[set[str]] = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        comp: set[str] = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(adj.get(x, ()))
        components.append(comp)

    isolates: list[Isolate] = []
    idx = 0
    for n in sorted(nodes):
        deg = degree.get(n, 0)
        singleton = any(len(c) == 1 and n in c for c in components)
        if deg == 0 or singleton:
            isolates.append(
                Isolate(
                    id=f"graph_{idx}_{n}",
                    kind=IsolateKind.GRAPH,
                    label=n,
                    source="rule",
                    metadata={
                        "degree": deg,
                        "orphan": True,
                        "component_size": 1,
                    },
                )
            )
            idx += 1
        elif deg == 1:
            # Weakly attached — still report as soft isolate
            isolates.append(
                Isolate(
                    id=f"graph_{idx}_{n}",
                    kind=IsolateKind.GRAPH,
                    label=n,
                    source="rule",
                    metadata={
                        "degree": deg,
                        "orphan": False,
                        "leaf": True,
                    },
                )
            )
            idx += 1
    return isolates


def _soft_enrich(
    isolates: list[Isolate],
    *,
    text: str | None,
    backend: str,
) -> list[Isolate]:
    """Best-effort soft backends; never fail the rule path."""
    if backend in ("llmintent", "soft") and text:
        try:
            import llmintent  # noqa: F401
            for iso in isolates:
                iso.metadata.setdefault("soft_backend", "llmintent_available")
        except Exception:
            pass
    if backend in ("hf", "soft"):
        try:
            import transformers  # noqa: F401
            for iso in isolates:
                iso.metadata.setdefault("hf_available", True)
        except Exception:
            pass
    return isolates
