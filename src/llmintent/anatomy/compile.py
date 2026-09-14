"""Compile ordinary English onto the closed LLM-region catalogue.

Fly-brain lesson: map by *intent documents*, not catalogue wording. Exact
aliases still win. Unmatched English is dropped, not hashed onto a region.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from llmintent.anatomy.atlas import REGIONS, REGION_BY_ID, Region
from llmintent.anatomy.lexical import LexicalSpace

_CLAUSE = re.compile(
    r"(?<=[.!?;])\s+|\n+|\s*(?:\bthen\b|\bafter\b|\band then\b)\s+",
    re.I,
)
_SOFT_FLOOR = 0.18


@dataclass(frozen=True)
class CompileHit:
    region: str
    score: float
    span: str
    via: str  # alias | intent
    handles: str

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "score": round(self.score, 4),
            "span": self.span,
            "via": self.via,
            "handles": self.handles,
        }


@dataclass
class RegionPlan:
    text: str
    hits: list[CompileHit] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def regions(self) -> list[str]:
        seen: list[str] = []
        for h in self.hits:
            if h.region not in seen:
                seen.append(h.region)
        return seen

    def score(self, region_id: str) -> float:
        return max((h.score for h in self.hits if h.region == region_id), default=0.0)

    def occupancy(self) -> dict[str, float]:
        occ = {r.id: 0.0 for r in REGIONS}
        for h in self.hits:
            occ[h.region] = max(occ[h.region], h.score)
        return occ

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "hits": [h.to_dict() for h in self.hits],
            "regions": self.regions,
            "dropped": list(self.dropped),
            "occupancy": {k: round(v, 4) for k, v in self.occupancy().items() if v > 0},
            "notes": list(self.notes),
        }


@lru_cache(maxsize=1)
def _space() -> tuple[LexicalSpace, dict[str, object]]:
    docs = []
    for r in REGIONS:
        docs.append(r.intent)
        docs.append(r.probe)
        docs.append(" ".join(r.aliases))
        docs.append(r.handles)
    space = LexicalSpace().fit(docs)
    mat = {r.id: space.encode(r.intent + " " + r.probe + " " + " ".join(r.aliases)) for r in REGIONS}
    return space, mat


def _alias_table() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for r in REGIONS:
        rows.append((r.id.replace("_", " "), r.id))
        for a in r.aliases:
            rows.append((a.lower(), r.id))
    rows.sort(key=lambda t: -len(t[0]))
    return rows


def _alias_hits(clause: str) -> list[CompileHit]:
    q = (clause or "").lower()
    hits: list[CompileHit] = []
    seen: set[str] = set()
    for alias, rid in _alias_table():
        if not alias or rid in seen:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q):
            region = REGION_BY_ID[rid]
            seen.add(rid)
            hits.append(
                CompileHit(
                    region=rid,
                    score=1.0,
                    span=alias,
                    via="alias",
                    handles=region.handles,
                )
            )
    return hits


def _intent_hit(clause: str, *, floor: float = _SOFT_FLOOR) -> CompileHit | None:
    space, mat = _space()
    q = space.encode(clause)
    best_id = ""
    best = -1.0
    second = -1.0
    for rid, vec in mat.items():
        s = float(q @ vec)
        if s > best:
            second = best
            best = s
            best_id = rid
        elif s > second:
            second = s
    margin = best - max(second, 0.0)
    if best < floor or margin < 0.02:
        return None
    region: Region = REGION_BY_ID[best_id]
    return CompileHit(
        region=best_id,
        score=best,
        span=clause.strip(),
        via="intent",
        handles=region.handles,
    )


def compile_regions(text: str, *, floor: float = _SOFT_FLOOR) -> RegionPlan:
    """Map text onto atlas regions. Unmatched clauses are dropped."""
    raw = (text or "").strip()
    plan = RegionPlan(text=raw)
    if not raw:
        plan.notes.append("Empty input.")
        return plan

    parts = [p.strip() for p in _CLAUSE.split(raw) if p and p.strip()]
    if not parts:
        parts = [raw]

    for clause in parts:
        aliases = _alias_hits(clause)
        intent = _intent_hit(clause, floor=floor)
        chosen: list[CompileHit] = list(aliases)
        alias_regions = {h.region for h in aliases}
        if intent and intent.region not in alias_regions:
            chosen.append(intent)
        if not chosen:
            plan.dropped.append(clause)
            continue
        plan.hits.extend(chosen)

    if not plan.hits:
        plan.notes.append("No region cleared the intent floor; nothing was hashed.")
    plan.notes.append(
        "Compile is a closed-catalogue prior, not a claim the model contains these neuropils."
    )
    return plan
