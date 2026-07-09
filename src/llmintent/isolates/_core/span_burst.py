"""Span isolates and creative-burst hopping along text spans.

A :class:`~intentisolates.types.SpanIsolate` is a typed isolate bound to a
contiguous text span. :class:`CreativeBurstHopper` walks span→span under
``linear``, ``motif_jump``, or ``creative_burst`` modes so divergent
exploration can still revisit goal / constraint anchors.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Any, Sequence

from llmintent.isolates._core.identify import identify_isolates
from llmintent.isolates._core.layers import layer_name_for
from llmintent.isolates._core.motifs import form_motifs
from llmintent.isolates._core.types import (
    ABSTRACT_LAYERS,
    BurstHop,
    BurstPath,
    Isolate,
    Motif,
    SpanIsolate,
    TextSpan,
    TypologyLabel,
)

# Typologies treated as structural anchors (protect + higher hop weight).
_ANCHOR_TYPS = frozenset(
    {
        TypologyLabel.GOAL.value,
        TypologyLabel.CONSTRAINT.value,
        TypologyLabel.OUTCOME.value,
    }
)

# Soft affinity priors by typology (creative_burst prefers affective / novel bridges).
_AFFINITY_PRIOR: dict[str, float] = {
    TypologyLabel.AFFECTIVE.value: 1.0,
    TypologyLabel.LEXICAL.value: 0.85,
    TypologyLabel.LATENT_FEATURE.value: 0.80,
    TypologyLabel.INSTRUMENTAL.value: 0.75,
    TypologyLabel.ACTION.value: 0.65,
    TypologyLabel.OUTCOME.value: 0.50,
    TypologyLabel.GOAL.value: 0.45,
    TypologyLabel.CONSTRAINT.value: 0.35,
    TypologyLabel.CONFOUNDER.value: 0.30,
    TypologyLabel.ORPHAN_NODE.value: 0.25,
    TypologyLabel.NOISE.value: 0.15,
    TypologyLabel.UNKNOWN.value: 0.55,
}

_HOP_WEIGHT_PRIOR: dict[str, float] = {
    TypologyLabel.GOAL.value: 1.4,
    TypologyLabel.CONSTRAINT.value: 1.35,
    TypologyLabel.OUTCOME.value: 1.25,
    TypologyLabel.ACTION.value: 1.15,
    TypologyLabel.INSTRUMENTAL.value: 1.05,
    TypologyLabel.AFFECTIVE.value: 0.95,
    TypologyLabel.LATENT_FEATURE.value: 0.90,
    TypologyLabel.LEXICAL.value: 0.70,
    TypologyLabel.CONFOUNDER.value: 0.55,
    TypologyLabel.NOISE.value: 0.35,
    TypologyLabel.ORPHAN_NODE.value: 0.50,
    TypologyLabel.UNKNOWN.value: 0.80,
}


def _typ_str(v: Any) -> str:
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


def identify_span_isolates(
    text: str,
    *,
    backend: str = "rule",
    assign_layer: bool = True,
    layer_strategy: str = "abstract",
    **kwargs: Any,
) -> list[SpanIsolate]:
    """Identify :class:`SpanIsolate` units from free text (offline rule path).

    Wraps :func:`~intentisolates.identify.identify_isolates` and binds each
    text isolate to a :class:`TextSpan` with ``hop_weight`` / ``burst_affinity``.
    """
    text = (text or "").strip()
    if not text:
        return []

    isolates = identify_isolates(
        text=text,
        backend=backend,
        assign_layer=assign_layer,
        layer_strategy=layer_strategy,
        **kwargs,
    )
    return span_isolates_from_isolates(isolates, text=text)


def span_isolates_from_isolates(
    isolates: Sequence[Isolate],
    *,
    text: str | None = None,
) -> list[SpanIsolate]:
    """Project plain :class:`Isolate` objects into span-bound hoppable units."""
    out: list[SpanIsolate] = []
    for i, iso in enumerate(isolates):
        kind = iso.kind.value if hasattr(iso.kind, "value") else str(iso.kind)
        # Prefer text isolates; keep feature/graph only when a span is already bound
        if kind != "text" and iso.span is None:
            continue

        typ = _typ_str(iso.typology)
        surface = iso.label or ""
        if iso.span is not None:
            start, end = int(iso.span[0]), int(iso.span[1])
            if text and 0 <= start < end <= len(text):
                surface = text[start:end]
        else:
            start = 0
            end = len(surface)
            if text and surface:
                found = text.find(surface)
                if found >= 0:
                    start, end = found, found + len(surface)

        sent_idx = iso.metadata.get("sentence_index")
        if sent_idx is None and text:
            sent_idx = text[: max(start, 0)].count(".") + text[: max(start, 0)].count("!") + text[
                : max(start, 0)
            ].count("?")

        hop_w = float(_HOP_WEIGHT_PRIOR.get(typ, 0.8))
        affinity = float(_AFFINITY_PRIOR.get(typ, 0.4))
        # Confidence + creative-cue boosts
        affinity = min(1.0, affinity + 0.15 * float(iso.confidence or 0.0))
        if re.search(
            r"(?i)\b(imagine|spark|wild|dream|color|texture|rhythm|unexpected|"
            r"metaphor|twist|burst|playful|curious)\b",
            surface,
        ):
            affinity = min(1.0, affinity + 0.12)
        protect = typ in _ANCHOR_TYPS

        layer = iso.layer
        layer_name = iso.layer_name
        if layer is None:
            layer = 2
            layer_name = ABSTRACT_LAYERS.get(2, "L2_latent_workspace")
        elif not layer_name:
            layer_name = layer_name_for(layer)

        out.append(
            SpanIsolate(
                id=iso.id if iso.id else f"span_{i}",
                typology=iso.typology,
                text_span=TextSpan(
                    start=start,
                    end=end,
                    surface=surface,
                    sentence_index=int(sent_idx) if sent_idx is not None else None,
                ),
                layer=layer,
                layer_name=layer_name,
                hop_weight=round(hop_w, 3),
                burst_affinity=round(affinity, 3),
                confidence=float(iso.confidence or 0.0),
                rationale=iso.rationale or f"Span isolate ({typ})",
                source=iso.source or "rule",
                protect=protect,
                metadata={
                    **dict(iso.metadata),
                    "from_isolate_id": iso.id,
                },
            )
        )
    return out


# Default knobs for creative_burst_v2 (reasoning-trace oriented).
_V2_DEFAULTS: dict[str, float] = {
    "anchor_pull": 0.70,
    "layer_bias": 0.55,
    "novelty_weight": 1.10,
    "motif_weight": 0.45,
    "anchor_schedule": 3.0,
    "side_hop_prob": 0.18,
}


class CreativeBurstHopper:
    """Hop span→span for linear walk, motif jumps, or creative bursts.

    Modes
    -----
    ``linear``
        Next unused span in document order (start offset).
    ``motif_jump``
        Prefer a co-member of a shared motif, else adjacent layer.
    ``creative_burst``
        v1 scoring: novelty, absolute layer jump, affinity, soft anchor pull,
        and a periodic forced-anchor pulse (every 2 hops).
    ``creative_burst_v2``
        Hybrid for reasoning traces: α·novelty + β·motif_link + γ·anchor_need
        + δ·layer_progress, with scheduled protect-span visits and occasional
        creative side-hops. See docs/CREATIVE_BURST_IMPROVEMENTS.md.
    ``random``
        Uniform among unvisited (baseline for experiments).
    """

    MODES = ("linear", "motif_jump", "creative_burst", "creative_burst_v2", "random")

    def __init__(
        self,
        spans: Sequence[SpanIsolate],
        *,
        motifs: Sequence[Motif] | None = None,
        seed: int = 17,
        anchor_pull: float | None = None,
        layer_bias: float = 0.55,
        novelty_weight: float = 1.10,
        motif_weight: float = 0.45,
        anchor_schedule: int = 3,
        side_hop_prob: float = 0.18,
    ) -> None:
        self.spans = list(spans)
        self.by_id = {s.id: s for s in self.spans}
        self.ordered = sorted(self.spans, key=lambda s: (s.start, s.id))
        self.motifs = list(motifs) if motifs is not None else []
        if motifs is None and self.spans:
            # Soft motifs from projected isolates
            self.motifs = form_motifs([s.to_isolate() for s in self.spans])
        self.seed = seed
        # Explicit None → v1-compatible 0.55; v2 mode applies _V2_DEFAULTS["anchor_pull"] at score time
        # when ``_v2_anchor_explicit`` is False.
        self._v2_anchor_explicit = anchor_pull is not None
        self.anchor_pull = 0.55 if anchor_pull is None else float(anchor_pull)
        self.layer_bias = float(layer_bias)
        self.novelty_weight = float(novelty_weight)
        self.motif_weight = float(motif_weight)
        self.anchor_schedule = max(0, int(anchor_schedule))
        self.side_hop_prob = min(1.0, max(0.0, float(side_hop_prob)))
        self._motif_neighbors = self._build_motif_neighbors()

    def knobs(self) -> dict[str, float]:
        """Return current scoring knobs (useful for experiment metadata)."""
        return {
            "anchor_pull": self.anchor_pull,
            "layer_bias": self.layer_bias,
            "novelty_weight": self.novelty_weight,
            "motif_weight": self.motif_weight,
            "anchor_schedule": float(self.anchor_schedule),
            "side_hop_prob": self.side_hop_prob,
        }

    @classmethod
    def for_v2(
        cls,
        spans: Sequence[SpanIsolate],
        *,
        motifs: Sequence[Motif] | None = None,
        seed: int = 17,
        **overrides: Any,
    ) -> CreativeBurstHopper:
        """Construct a hopper with ``creative_burst_v2`` recommended defaults."""
        kwargs = dict(_V2_DEFAULTS)
        kwargs["anchor_schedule"] = int(kwargs["anchor_schedule"])
        kwargs.update(overrides)
        return cls(spans, motifs=motifs, seed=seed, **kwargs)

    def _build_motif_neighbors(self) -> dict[str, set[str]]:
        nbrs: dict[str, set[str]] = {s.id: set() for s in self.spans}
        for m in self.motifs:
            members = [mid for mid in m.member_ids if mid in nbrs]
            for a in members:
                for b in members:
                    if a != b:
                        nbrs[a].add(b)
        return nbrs

    def hop(
        self,
        current_span_id: str,
        *,
        mode: str = "creative_burst",
        visited: Sequence[str] | None = None,
    ) -> BurstHop:
        """Choose the next span from ``current_span_id`` under ``mode``."""
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {self.MODES}")
        if current_span_id not in self.by_id:
            raise KeyError(f"Unknown span id: {current_span_id}")
        current = self.by_id[current_span_id]
        visited_list = list(visited) if visited is not None else [current_span_id]
        rng = random.Random(self.seed + hash(current_span_id) % 10_000 + len(visited_list))
        nxt, score, reason = self._next_hop(current, visited_list, mode=mode, rng=rng)
        if nxt is None:
            return BurstHop(
                from_id=current_span_id,
                to_id=current_span_id,
                mode=mode,
                score=0.0,
                reason="exhausted",
            )
        return BurstHop(
            from_id=current_span_id,
            to_id=nxt.id,
            mode=mode,
            score=round(score, 4),
            reason=reason,
        )

    def burst_path(
        self,
        seed: str | int | SpanIsolate | None = None,
        n_hops: int = 5,
        *,
        mode: str = "creative_burst",
    ) -> BurstPath:
        """Return an ordered hop path of length up to ``n_hops`` from ``seed``."""
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {self.MODES}")
        if not self.spans:
            return BurstPath(seed_id="", hops=[], span_ids=[], typology_path=[], mode=mode, summary="empty")

        start = self._resolve_seed(seed)
        rng = random.Random(self.seed + hash(start.id) % 10_000)

        visited: list[str] = [start.id]
        hops: list[BurstHop] = []
        current = start

        for _ in range(max(0, n_hops)):
            nxt, score, reason = self._next_hop(current, visited, mode=mode, rng=rng)
            if nxt is None:
                break
            hops.append(
                BurstHop(
                    from_id=current.id,
                    to_id=nxt.id,
                    mode=mode,
                    score=round(score, 4),
                    reason=reason,
                )
            )
            visited.append(nxt.id)
            current = nxt

        typ_path = [_typ_str(self.by_id[i].typology) for i in visited if i in self.by_id]
        summary = (
            f"{mode}: {len(hops)} hops from `{start.id}` -> "
            + " -> ".join(typ_path)
        )
        return BurstPath(
            seed_id=start.id,
            hops=hops,
            span_ids=visited,
            typology_path=typ_path,
            mode=mode,
            summary=summary,
            metadata={
                "n_spans": len(self.spans),
                "n_motifs": len(self.motifs),
                "seed": self.seed,
                **self.knobs(),
            },
        )

    def _resolve_seed(self, seed: str | int | SpanIsolate | None) -> SpanIsolate:
        if isinstance(seed, SpanIsolate):
            return seed
        if isinstance(seed, int):
            if 0 <= seed < len(self.ordered):
                return self.ordered[seed]
            raise IndexError(f"seed index {seed} out of range (n={len(self.ordered)})")
        if isinstance(seed, str) and seed in self.by_id:
            return self.by_id[seed]
        # Prefer a goal/constraint seed when unspecified
        for s in self.ordered:
            if _typ_str(s.typology) in _ANCHOR_TYPS:
                return s
        return self.ordered[0]

    def _next_hop(
        self,
        current: SpanIsolate,
        visited: Sequence[str],
        *,
        mode: str,
        rng: random.Random,
    ) -> tuple[SpanIsolate | None, float, str]:
        visited_set = set(visited)
        candidates = [s for s in self.spans if s.id not in visited_set]
        if not candidates:
            return None, 0.0, "exhausted"

        if mode == "linear":
            # Next in document order after current
            after = [s for s in self.ordered if s.id not in visited_set and s.start >= current.end]
            pick = after[0] if after else next(s for s in self.ordered if s.id not in visited_set)
            return pick, 1.0, "linear document order"

        if mode == "random":
            pick = rng.choice(candidates)
            return pick, 1.0 / len(candidates), "uniform random"

        if mode == "motif_jump":
            motif_cands = [s for s in candidates if s.id in self._motif_neighbors.get(current.id, ())]
            pool = motif_cands or candidates
            # Prefer similar/adjacent layer among motif neighbors
            scored = []
            for s in pool:
                dlayer = abs(_layer_int(s.layer) - _layer_int(current.layer))
                score = s.hop_weight / (1.0 + 0.4 * dlayer)
                if s.id in self._motif_neighbors.get(current.id, ()):
                    score += 0.5
                scored.append((score, s))
            scored.sort(key=lambda x: (-x[0], x[1].id))
            best_score, pick = scored[0]
            reason = "motif co-member" if motif_cands else "motif fallback (layer-adjacent)"
            return pick, best_score, reason

        if mode == "creative_burst_v2":
            return self._next_hop_v2(current, visited, candidates, visited_set, rng=rng)

        # creative_burst (v1)
        visited_typs = {_typ_str(self.by_id[i].typology) for i in visited if i in self.by_id}
        anchor_ids = [s.id for s in self.spans if s.protect]
        unvisited_anchors = [s for s in candidates if s.id in set(anchor_ids)]
        visited_anchors = sum(1 for a in anchor_ids if a in visited_set)
        need_anchor = visited_anchors < min(2, len(anchor_ids)) and len(visited) >= 2
        # Periodic forced visit so anchors stay in the creative path
        if unvisited_anchors and (len(visited) % 2 == 0):
            pick = max(unvisited_anchors, key=lambda s: (s.hop_weight, s.id))
            return pick, 5.0 + pick.hop_weight, "forced_anchor_pulse"

        scored: list[tuple[float, SpanIsolate, str]] = []
        for s in candidates:
            typ = _typ_str(s.typology)
            novelty = 1.0 if typ not in visited_typs else 0.25
            dlayer = abs(_layer_int(s.layer) - _layer_int(current.layer))
            # Prefer moderate layer jumps (creative) over pure adjacency
            layer_term = 0.35 * min(dlayer, 3) + 0.15 * (1.0 if dlayer >= 2 else 0.0)
            affinity = s.burst_affinity * s.hop_weight
            motif_bonus = 0.4 if s.id in self._motif_neighbors.get(current.id, ()) else 0.0
            # Soft pull toward anchors if under-visited (stronger than random baseline)
            anchor_term = 0.0
            if need_anchor and s.protect:
                anchor_term = self.anchor_pull * 1.8
            elif s.protect:
                anchor_term = self.anchor_pull * 0.75
            # Distance: avoid always picking nearest (encourage burst)
            gap = abs(s.start - current.start)
            dist_term = 0.2 * math.log1p(gap / 40.0)
            score = novelty + layer_term + affinity + motif_bonus + anchor_term + dist_term
            # Tiny jitter for tie-breaks (deterministic via rng)
            score += 0.01 * rng.random()
            reason_bits = []
            if novelty >= 1.0:
                reason_bits.append("novel_typology")
            if dlayer >= 2:
                reason_bits.append("layer_jump")
            if motif_bonus:
                reason_bits.append("motif")
            if anchor_term > 0.3:
                reason_bits.append("anchor_pull")
            scored.append((score, s, "+".join(reason_bits) or "affinity"))

        scored.sort(key=lambda x: (-x[0], x[1].id))
        best_score, pick, reason = scored[0]
        return pick, best_score, reason

    def _next_hop_v2(
        self,
        current: SpanIsolate,
        visited: Sequence[str],
        candidates: list[SpanIsolate],
        visited_set: set[str],
        *,
        rng: random.Random,
    ) -> tuple[SpanIsolate | None, float, str]:
        """Hybrid novelty + motif + anchor + forward-layer scoring."""
        alpha = self.novelty_weight
        beta = self.motif_weight
        gamma = (
            self.anchor_pull
            if self._v2_anchor_explicit
            else float(_V2_DEFAULTS["anchor_pull"])
        )
        delta = self.layer_bias
        schedule = self.anchor_schedule

        visited_typs = {_typ_str(self.by_id[i].typology) for i in visited if i in self.by_id}
        anchor_ids = [s.id for s in self.spans if s.protect]
        unvisited_anchors = [s for s in candidates if s.protect]
        visited_anchors = sum(1 for a in anchor_ids if a in visited_set)
        n_visited = len(visited)
        need_anchor = visited_anchors < min(2, len(anchor_ids)) and n_visited >= 2

        # Anchor-scheduled burst: periodic forced protect visit
        if (
            schedule > 0
            and unvisited_anchors
            and n_visited >= 2
            and (n_visited % schedule == 0)
        ):
            # Prefer goal/constraint over outcome when both unvisited
            pick = max(
                unvisited_anchors,
                key=lambda s: (
                    2 if _typ_str(s.typology) in ("goal", "constraint") else 1,
                    s.hop_weight,
                    s.id,
                ),
            )
            return pick, 6.0 + pick.hop_weight, "scheduled_anchor"

        # Soft urgency: if many hops left without anchors, raise need
        if unvisited_anchors and visited_anchors == 0 and n_visited >= 3:
            need_anchor = True

        side_hop = rng.random() < self.side_hop_prob
        cur_layer = _layer_int(current.layer)

        scored: list[tuple[float, SpanIsolate, str]] = []
        for s in candidates:
            typ = _typ_str(s.typology)
            novelty = 1.0 if typ not in visited_typs else 0.22
            # Mild extra novelty for affective / lexical bridges
            if typ in (
                TypologyLabel.AFFECTIVE.value,
                TypologyLabel.LEXICAL.value,
                TypologyLabel.LATENT_FEATURE.value,
            ) and typ not in visited_typs:
                novelty += 0.12

            s_layer = _layer_int(s.layer)
            d_forward = s_layer - cur_layer
            if side_hop:
                # Creative side-hop: reward moderate absolute jumps, ignore forward bias
                layer_term = 0.25 * min(abs(d_forward), 3)
            else:
                # Prefer forward layer progress; soft penalty for large backward jumps
                layer_term = delta * max(0.0, float(d_forward))
                if d_forward < -1:
                    layer_term -= 0.25 * min(abs(d_forward), 3)
                # Small bonus for staying same layer when exploring motifs
                if d_forward == 0:
                    layer_term += 0.08

            affinity = 0.55 * s.burst_affinity * (0.5 + 0.5 * s.hop_weight)
            motif_link = beta if s.id in self._motif_neighbors.get(current.id, ()) else 0.0

            anchor_need = 0.0
            if s.protect:
                if need_anchor:
                    anchor_need = gamma * 2.0
                else:
                    # Keep soft pull so anchors remain competitive without dominating
                    anchor_need = gamma * 0.85
                if _typ_str(s.typology) in ("goal", "constraint"):
                    anchor_need += 0.15 * gamma

            gap = abs(s.start - current.start)
            dist_term = 0.15 * math.log1p(gap / 50.0)

            score = (
                alpha * novelty
                + motif_link
                + anchor_need
                + layer_term
                + affinity
                + dist_term
                + 0.01 * rng.random()
            )
            reason_bits = []
            if novelty >= 1.0:
                reason_bits.append("novel_typology")
            if motif_link > 0:
                reason_bits.append("motif")
            if anchor_need > 0.4:
                reason_bits.append("anchor_need")
            if not side_hop and d_forward > 0:
                reason_bits.append("layer_forward")
            if side_hop:
                reason_bits.append("side_hop")
            scored.append((score, s, "+".join(reason_bits) or "affinity"))

        scored.sort(key=lambda x: (-x[0], x[1].id))
        best_score, pick, reason = scored[0]
        return pick, best_score, reason


def typology_path_entropy(typology_path: Sequence[str]) -> float:
    """Shannon entropy (bits) of typology labels along a burst path."""
    if not typology_path:
        return 0.0
    counts: dict[str, int] = {}
    for t in typology_path:
        counts[t] = counts.get(t, 0) + 1
    n = len(typology_path)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return round(ent, 4)


def layer_path_monotonicity(
    spans: Sequence[SpanIsolate],
    span_ids: Sequence[str],
) -> float:
    """Fraction of consecutive hops with non-decreasing abstract layer.

    Forward or same-layer steps count as monotonic; only strict backward
    layer moves hurt the score. Empty / single-node paths return 1.0.
    """
    by_id = {s.id: s for s in spans}
    layers = [_layer_int(by_id[i].layer) for i in span_ids if i in by_id]
    if len(layers) < 2:
        return 1.0
    ok = sum(1 for a, b in zip(layers, layers[1:]) if b >= a)
    return round(ok / (len(layers) - 1), 4)


def filter_spans_for_burst(
    spans: Sequence[SpanIsolate],
    *,
    protect_only: bool = False,
    hot_ids: Sequence[str] | None = None,
    drop_noise: bool = True,
) -> list[SpanIsolate]:
    """Filter spans for post-compaction / protect-hot-set bursting.

    Use after isolate-then-compact: keep ``protect`` anchors and optional
    ``hot_ids`` from the working set, optionally dropping noise typologies.
    """
    hot = set(hot_ids) if hot_ids is not None else None
    out: list[SpanIsolate] = []
    for s in spans:
        typ = _typ_str(s.typology)
        if drop_noise and typ in (TypologyLabel.NOISE.value, TypologyLabel.CONFOUNDER.value):
            if not s.protect:
                continue
        if protect_only and not s.protect:
            continue
        if hot is not None and s.id not in hot and not s.protect:
            continue
        out.append(s)
    return out


def burst_path_from_text(
    text: str,
    *,
    seed: str | int | None = None,
    n_hops: int = 5,
    mode: str = "creative_burst",
    rng_seed: int = 17,
    backend: str = "rule",
) -> tuple[list[SpanIsolate], BurstPath]:
    """Convenience: identify span isolates and run a burst path."""
    spans = identify_span_isolates(text, backend=backend)
    if mode == "creative_burst_v2":
        hopper = CreativeBurstHopper.for_v2(spans, seed=rng_seed)
    else:
        hopper = CreativeBurstHopper(spans, seed=rng_seed)
    path = hopper.burst_path(seed=seed, n_hops=n_hops, mode=mode)
    return spans, path


def multi_path_burst(
    spans: Sequence[SpanIsolate],
    *,
    n_hops: int = 5,
    mode: str = "creative_burst_v2",
    k: int = 5,
    seed: int = 17,
    select_by: str = "tradeoff_harmonic",
    hopper_kwargs: dict[str, Any] | None = None,
) -> tuple[BurstPath, list[dict[str, Any]]]:
    """ToT/GoT-style: run ``k`` burst paths and pick the best by CreativityMeter.

    ``select_by`` one of: ``tradeoff_harmonic``, ``tradeoff_product``,
    ``creativity_score``, ``reasoning_trace_score``, ``anchor_visit_rate``.
    """
    from llmintent.isolates._core.creativity import CreativityMeter

    kwargs = dict(hopper_kwargs or {})
    if mode == "creative_burst_v2" and not kwargs:
        hopper = CreativeBurstHopper.for_v2(spans, seed=seed)
    else:
        hopper = CreativeBurstHopper(spans, seed=seed, **kwargs)

    meter = CreativityMeter()
    candidates: list[dict[str, Any]] = []
    n_seeds = max(1, min(k, len(spans)))
    for i in range(n_seeds):
        h = CreativeBurstHopper(
            spans,
            motifs=hopper.motifs,
            seed=seed + i * 31,
            anchor_pull=hopper.anchor_pull if hopper._v2_anchor_explicit or mode != "creative_burst_v2" else None,
            layer_bias=hopper.layer_bias,
            novelty_weight=hopper.novelty_weight,
            motif_weight=hopper.motif_weight,
            anchor_schedule=hopper.anchor_schedule,
            side_hop_prob=hopper.side_hop_prob,
        )
        if mode == "creative_burst_v2" and not hopper._v2_anchor_explicit:
            h = CreativeBurstHopper.for_v2(
                spans,
                motifs=hopper.motifs,
                seed=seed + i * 31,
                layer_bias=hopper.layer_bias,
                novelty_weight=hopper.novelty_weight,
                motif_weight=hopper.motif_weight,
                anchor_schedule=hopper.anchor_schedule,
                side_hop_prob=hopper.side_hop_prob,
            )
        path = h.burst_path(seed=i % max(1, len(h.ordered)), n_hops=n_hops, mode=mode)
        report = meter.score_burst(path, spans, motif_neighbors=h._motif_neighbors)
        score_map = {
            "tradeoff_harmonic": report.tradeoff_harmonic,
            "tradeoff_product": report.tradeoff_product,
            "creativity_score": report.creativity_score,
            "reasoning_trace_score": report.reasoning_trace_score,
            "anchor_visit_rate": report.anchor_visit_rate,
        }
        if select_by not in score_map:
            raise ValueError(f"Unknown select_by={select_by!r}")
        candidates.append(
            {
                "path": path,
                "report": report,
                "select_score": score_map[select_by],
                "seed_index": i,
            }
        )

    candidates.sort(key=lambda c: (-float(c["select_score"]), c["seed_index"]))
    best = candidates[0]["path"]
    best.metadata = {
        **dict(best.metadata),
        "multi_path_k": n_seeds,
        "select_by": select_by,
        "select_score": candidates[0]["select_score"],
        "candidate_scores": [
            {
                "seed_index": c["seed_index"],
                "select_score": c["select_score"],
                "C": c["report"].creativity_score,
                "R": c["report"].reasoning_trace_score,
                "H": c["report"].tradeoff_harmonic,
            }
            for c in candidates
        ],
    }
    return best, candidates
