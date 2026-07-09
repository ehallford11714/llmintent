"""Creativity meter for span-burst paths and isolate sets.

Maps Guilford-style divergent-thinking dimensions (fluency, flexibility,
originality/novelty, elaboration) plus constraint fidelity onto hop paths so
creative exploration can be scored *with* reasoning-trace anchors.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence

from llmintent.isolates._core.span_burst import (
    layer_path_monotonicity,
    typology_path_entropy,
)
from llmintent.isolates._core.types import BurstPath, CreativityReport, SpanIsolate

_ANCHOR_TYPS = frozenset({"goal", "constraint", "outcome"})


def _typ(v: Any) -> str:
    return v.value if hasattr(v, "value") else str(v)


def _layer_int(layer: int | str | None) -> int:
    if layer is None:
        return 2
    if isinstance(layer, int):
        return layer
    s = str(layer)
    if s.isdigit():
        return int(s)
    m = re.match(r"L(\d+)", s, re.I)
    return int(m.group(1)) if m else 2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _harmonic(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


class CreativityMeter:
    """Score hop paths / span sets on creativity and reasoning-trace quality.

    Dimensions (≈ Guilford + constrained creativity)
    -----------------------------------------------
    diversity
        Normalized typology Shannon entropy along the path.
    novelty
        Fraction of typologies (and optional surfaces) unseen vs a prior /
        corpus baseline; falls back to within-path first-occurrence rate.
    flexibility
        Cross-layer and cross-motif jump rate (category switching).
    elaboration
        Mean surface length / detail proxy (log-scaled).
    fluency
        Unique spans visited relative to available pool (idea count proxy).
    constraint_fidelity
        Protect / goal-constraint visit rate (creativity *with* anchors).

    Composites
    ----------
    creativity_score (C)
        Weighted mean of diversity, novelty, flexibility, elaboration, fluency.
    reasoning_trace_score (R)
        Weighted mean of constraint_fidelity and layer_monotonicity.
    tradeoff_product / tradeoff_harmonic
        C·R and harmonic mean for Pareto-style selection.
    """

    def __init__(
        self,
        *,
        w_diversity: float = 0.25,
        w_novelty: float = 0.25,
        w_flexibility: float = 0.20,
        w_elaboration: float = 0.15,
        w_fluency: float = 0.15,
        w_constraint: float = 0.65,
        w_layer_mono: float = 0.35,
        prior_typologies: Sequence[str] | None = None,
        prior_surfaces: Sequence[str] | None = None,
        max_entropy_bits: float = 3.0,
        elaboration_ref_chars: float = 80.0,
    ) -> None:
        self.w_diversity = w_diversity
        self.w_novelty = w_novelty
        self.w_flexibility = w_flexibility
        self.w_elaboration = w_elaboration
        self.w_fluency = w_fluency
        self.w_constraint = w_constraint
        self.w_layer_mono = w_layer_mono
        self.prior_typologies = set(prior_typologies or ())
        self.prior_surfaces = {s.strip().lower() for s in (prior_surfaces or ()) if s}
        self.max_entropy_bits = max(1e-6, float(max_entropy_bits))
        self.elaboration_ref_chars = max(1.0, float(elaboration_ref_chars))

    def score_path(
        self,
        path_spans: Sequence[SpanIsolate] | BurstPath,
        *,
        all_spans: Sequence[SpanIsolate] | None = None,
        motif_neighbors: dict[str, set[str]] | None = None,
    ) -> CreativityReport:
        """Score an ordered path of spans or a :class:`BurstPath`."""
        if isinstance(path_spans, BurstPath):
            if all_spans is None:
                raise ValueError("score_path(BurstPath) requires all_spans=")
            by_id = {s.id: s for s in all_spans}
            spans = [by_id[i] for i in path_spans.span_ids if i in by_id]
            span_ids = list(path_spans.span_ids)
            pool = list(all_spans)
        else:
            spans = list(path_spans)
            span_ids = [s.id for s in spans]
            pool = list(all_spans) if all_spans is not None else spans
        return self._score(spans, span_ids=span_ids, pool=pool, motif_neighbors=motif_neighbors)

    def score_burst(
        self,
        path: BurstPath,
        spans: Sequence[SpanIsolate],
        *,
        motif_neighbors: dict[str, set[str]] | None = None,
    ) -> CreativityReport:
        """Alias for scoring a burst path against its span pool."""
        return self.score_path(path, all_spans=spans, motif_neighbors=motif_neighbors)

    def score_spans(
        self,
        spans: Sequence[SpanIsolate],
        *,
        motif_neighbors: dict[str, set[str]] | None = None,
    ) -> CreativityReport:
        """Score an unordered / document-order span set as if it were a path."""
        ordered = sorted(spans, key=lambda s: (s.start, s.id))
        return self._score(
            ordered,
            span_ids=[s.id for s in ordered],
            pool=list(spans),
            motif_neighbors=motif_neighbors,
        )

    def score_text_burst(
        self,
        text: str,
        *,
        n_hops: int = 5,
        mode: str = "creative_burst_v2",
        seed: int = 17,
    ) -> tuple[list[SpanIsolate], BurstPath, CreativityReport]:
        """Identify spans, hop, and meter in one call."""
        from llmintent.isolates._core.span_burst import CreativeBurstHopper, identify_span_isolates

        spans = identify_span_isolates(text)
        if mode == "creative_burst_v2":
            hopper = CreativeBurstHopper.for_v2(spans, seed=seed)
        else:
            hopper = CreativeBurstHopper(spans, seed=seed)
        path = hopper.burst_path(n_hops=n_hops, mode=mode)
        report = self.score_burst(path, spans, motif_neighbors=hopper._motif_neighbors)
        return spans, path, report

    def _score(
        self,
        spans: Sequence[SpanIsolate],
        *,
        span_ids: Sequence[str],
        pool: Sequence[SpanIsolate],
        motif_neighbors: dict[str, set[str]] | None,
    ) -> CreativityReport:
        if not spans:
            return CreativityReport(metadata={"empty": True})

        typs = [_typ(s.typology) for s in spans]
        ent = typology_path_entropy(typs)
        diversity = _clamp01(ent / self.max_entropy_bits)

        # Novelty vs prior corpus; else within-path first-seen rate
        if self.prior_typologies or self.prior_surfaces:
            novel_flags = []
            for s in spans:
                t = _typ(s.typology)
                surf = (s.surface or "").strip().lower()
                typ_novel = t not in self.prior_typologies if self.prior_typologies else True
                surf_novel = surf not in self.prior_surfaces if self.prior_surfaces else True
                novel_flags.append(1.0 if (typ_novel and surf_novel) else 0.0)
            novelty = sum(novel_flags) / len(novel_flags)
        else:
            seen: set[str] = set()
            firsts = 0
            for t in typs:
                if t not in seen:
                    firsts += 1
                    seen.add(t)
            novelty = firsts / max(1, len(typs))

        # Flexibility: layer changes + motif non-edges (category / structure switches)
        layers = [_layer_int(s.layer) for s in spans]
        layer_switches = 0
        motif_switches = 0
        n_edges = max(1, len(spans) - 1)
        nbrs = motif_neighbors or {}
        for i in range(len(spans) - 1):
            if layers[i + 1] != layers[i]:
                layer_switches += 1
            a, b = spans[i].id, spans[i + 1].id
            if b not in nbrs.get(a, ()):
                motif_switches += 1
        flexibility = _clamp01(0.55 * (layer_switches / n_edges) + 0.45 * (motif_switches / n_edges))

        # Elaboration: mean surface length vs reference
        lengths = [len((s.surface or "").strip()) for s in spans]
        mean_len = sum(lengths) / max(1, len(lengths))
        elaboration = _clamp01(math.log1p(mean_len) / math.log1p(self.elaboration_ref_chars))

        # Fluency: unique spans / pool size (capped)
        unique = len(set(span_ids))
        fluency = _clamp01(unique / max(1, min(len(pool), max(unique, 1))))
        # Prefer coverage of pool when pool known
        if pool:
            fluency = _clamp01(unique / max(1, len(pool)))

        # Constraint fidelity
        anchors = [s for s in pool if s.protect or _typ(s.typology) in _ANCHOR_TYPS]
        anchor_ids = {s.id for s in anchors}
        visited_anchors = {i for i in span_ids if i in anchor_ids}
        if anchor_ids:
            constraint_fidelity = len(visited_anchors) / len(anchor_ids)
        else:
            constraint_fidelity = 1.0

        mono = layer_path_monotonicity(pool if pool else spans, span_ids)

        # Composites
        c_w = (
            self.w_diversity
            + self.w_novelty
            + self.w_flexibility
            + self.w_elaboration
            + self.w_fluency
        )
        creativity = (
            self.w_diversity * diversity
            + self.w_novelty * novelty
            + self.w_flexibility * flexibility
            + self.w_elaboration * elaboration
            + self.w_fluency * fluency
        ) / max(1e-6, c_w)

        r_w = self.w_constraint + self.w_layer_mono
        reasoning = (
            self.w_constraint * constraint_fidelity + self.w_layer_mono * mono
        ) / max(1e-6, r_w)

        product = creativity * reasoning
        harm = _harmonic(creativity, reasoning)

        return CreativityReport(
            diversity=round(diversity, 4),
            novelty=round(novelty, 4),
            flexibility=round(flexibility, 4),
            elaboration=round(elaboration, 4),
            constraint_fidelity=round(constraint_fidelity, 4),
            fluency=round(fluency, 4),
            layer_monotonicity=round(mono, 4),
            creativity_score=round(creativity, 4),
            reasoning_trace_score=round(reasoning, 4),
            tradeoff_product=round(product, 4),
            tradeoff_harmonic=round(harm, 4),
            n_spans_scored=len(spans),
            n_unique_typologies=len(set(typs)),
            typology_entropy=ent,
            anchor_visit_rate=round(constraint_fidelity, 4),
            metadata={
                "typology_counts": dict(Counter(typs)),
                "mean_surface_chars": round(mean_len, 2),
                "n_anchors_pool": len(anchor_ids),
                "n_anchors_visited": len(visited_anchors),
                "weights": {
                    "diversity": self.w_diversity,
                    "novelty": self.w_novelty,
                    "flexibility": self.w_flexibility,
                    "elaboration": self.w_elaboration,
                    "fluency": self.w_fluency,
                    "constraint": self.w_constraint,
                    "layer_mono": self.w_layer_mono,
                },
            },
        )
