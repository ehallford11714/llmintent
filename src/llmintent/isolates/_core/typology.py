"""Heuristic typology classification (always offline)."""

from __future__ import annotations

import re
from typing import Any

from llmintent.isolates._core.types import Isolate, IsolateKind, TypologyLabel

# Keyword / cue lexicons (lightweight, offline).
_CUES: dict[TypologyLabel, tuple[str, ...]] = {
    TypologyLabel.GOAL: (
        "want", "need", "goal", "aim", "intend", "purpose", "objective",
        "achieve", "should", "must", "plan to", "hope to", "try to",
    ),
    TypologyLabel.CONSTRAINT: (
        "cannot", "can't", "must not", "unless", "only if", "limit",
        "constraint", "restricted", "without", "except", "provided that",
        "budget", "deadline", "require",
    ),
    TypologyLabel.AFFECTIVE: (
        "feel", "afraid", "happy", "sad", "angry", "anxious", "love",
        "hate", "worry", "excited", "frustrated", "hope", "fear", "emotion",
    ),
    TypologyLabel.INSTRUMENTAL: (
        "using", "via", "through", "by means", "tool", "method", "with",
        "instrument", "leverage", "apply", "deploy", "via",
    ),
    TypologyLabel.ACTION: (
        "do", "make", "build", "run", "send", "call", "write", "create",
        "execute", "implement", "launch", "submit", "open", "close",
    ),
    TypologyLabel.OUTCOME: (
        "result", "outcome", "effect", "leads to", "causes", "therefore",
        "so that", "consequently", "yields", "produces",
    ),
    TypologyLabel.CONFOUNDER: (
        "confound", "spurious", "also correlated", "proxy for", "collider",
        "common cause", "selection bias",
    ),
    TypologyLabel.NOISE: (
        "um", "uh", "anyway", "whatever", "random", "noise", "filler",
    ),
}

_GOAL_RE = re.compile(
    r"\b(i want|i need|my goal|in order to|so that i|aim to)\b",
    re.I,
)
_CONSTRAINT_RE = re.compile(
    r"\b(cannot|can't|must not|as long as|only if|unless)\b",
    re.I,
)


def classify_typology(isolate: Isolate | str, **kwargs: Any) -> Isolate:
    """
    Classify an isolate (or raw string) into a TypologyLabel.

    Returns an Isolate with typology, confidence, and rationale filled in.
    Always works offline (rule/heuristic backend).
    """
    if isinstance(isolate, str):
        iso = Isolate(
            id=kwargs.get("id", "iso_0"),
            kind=IsolateKind.TEXT,
            label=isolate.strip(),
            layer=kwargs.get("layer"),
            layer_name=kwargs.get("layer_name"),
            source="rule",
        )
    else:
        iso = isolate

    text = (iso.label or "").strip().lower()
    if not text:
        iso.typology = TypologyLabel.NOISE
        iso.confidence = 0.4
        iso.rationale = "Empty or blank label → noise"
        return iso

    # Kind-based priors
    kind = iso.kind.value if hasattr(iso.kind, "value") else str(iso.kind)
    if kind == IsolateKind.GRAPH.value or kind == "graph":
        if iso.metadata.get("degree", 1) == 0 or iso.metadata.get("orphan"):
            iso.typology = TypologyLabel.ORPHAN_NODE
            iso.confidence = 0.9
            iso.rationale = "Graph node with no edges (orphan / causal isolate)"
            return iso
        if iso.metadata.get("confounder_hint"):
            iso.typology = TypologyLabel.CONFOUNDER
            iso.confidence = 0.75
            iso.rationale = "Graph metadata marks confounder-like role"
            return iso

    if kind == IsolateKind.FEATURE.value or kind == "feature":
        sparsity = float(iso.metadata.get("sparsity", 0.0))
        z = abs(float(iso.metadata.get("zscore", 0.0)))
        # Outlier / sparse dimensions → latent_feature hypothesis (SAE analogue).
        # Thresholds are intentionally moderate: callers already top-k filter.
        if sparsity >= 0.35 or z >= 1.5:
            iso.typology = TypologyLabel.LATENT_FEATURE
            iso.confidence = min(0.95, 0.5 + 0.12 * z + 0.25 * sparsity)
            iso.rationale = (
                f"Sparse/outlier feature (sparsity={sparsity:.2f}, |z|={z:.2f}) "
                "→ monosemantic-like latent_feature hypothesis"
            )
            return iso
        if z < 0.5 and sparsity < 0.2:
            iso.typology = TypologyLabel.NOISE
            iso.confidence = 0.6
            iso.rationale = "Dense, low-magnitude feature → noise prior"
            return iso
        # Mid-range feature still reported as latent_feature with lower confidence
        iso.typology = TypologyLabel.LATENT_FEATURE
        iso.confidence = 0.45
        iso.rationale = (
            f"Feature isolate without strong outlier score (|z|={z:.2f}); "
            "treated as weak latent_feature"
        )
        return iso

    scores: dict[TypologyLabel, float] = {t: 0.0 for t in TypologyLabel}
    hits: dict[TypologyLabel, list[str]] = {t: [] for t in TypologyLabel}

    for label, cues in _CUES.items():
        for cue in cues:
            if cue in text:
                scores[label] += 1.0 + 0.15 * len(cue.split())
                hits[label].append(cue)

    if _GOAL_RE.search(text):
        scores[TypologyLabel.GOAL] += 2.0
        hits[TypologyLabel.GOAL].append("goal_pattern")
    if _CONSTRAINT_RE.search(text):
        scores[TypologyLabel.CONSTRAINT] += 2.0
        hits[TypologyLabel.CONSTRAINT].append("constraint_pattern")

    # Layer prior: late layers favor goal/action; early favor lexical
    layer_idx = _as_int_layer(iso.layer)
    if layer_idx is not None:
        if layer_idx <= 1:
            scores[TypologyLabel.LEXICAL] += 0.8
        elif layer_idx == 2:
            scores[TypologyLabel.LATENT_FEATURE] += 0.5
        elif layer_idx == 3:
            scores[TypologyLabel.GOAL] += 0.4
            scores[TypologyLabel.CONSTRAINT] += 0.4
        elif layer_idx >= 4:
            scores[TypologyLabel.ACTION] += 0.5
            scores[TypologyLabel.OUTCOME] += 0.3

    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    if best_score <= 0:
        # Default: short phrases → lexical; else unknown
        if len(text.split()) <= 6:
            iso.typology = TypologyLabel.LEXICAL
            iso.confidence = 0.45
            iso.rationale = "No strong cues; short span treated as lexical isolate"
        else:
            iso.typology = TypologyLabel.UNKNOWN
            iso.confidence = 0.3
            iso.rationale = "No typology cues matched"
        return iso

    total = sum(scores.values()) or 1.0
    conf = min(0.95, 0.4 + 0.35 * (best_score / max(total, best_score)) + 0.05 * best_score)
    iso.typology = best
    iso.confidence = round(conf, 3)
    cue_str = ", ".join(hits[best][:5]) or "score"
    iso.rationale = f"Matched {best.value} cues ({cue_str}); score={best_score:.1f}"
    return iso


def _as_int_layer(layer: int | str | None) -> int | None:
    if layer is None:
        return None
    if isinstance(layer, int):
        return layer
    s = str(layer)
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    # L2_latent_workspace → 2
    m = re.match(r"L(\d+)", s, re.I)
    if m:
        return int(m.group(1))
    return None
