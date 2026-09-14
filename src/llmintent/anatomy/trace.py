"""How atlas occupancy varies through the spans of a prompt.

Compile each clause separately so a region card can say not only what it
does, but when it turns on, peaks, and drops along the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llmintent.anatomy.atlas import REGION_BY_ID, region_ids, what_region_does
from llmintent.anatomy.compile import compile_regions

# Keep causal conjunctions on the following span so "because" still compiles.
_SPAN = re.compile(
    r"(?<=[.!?;])\s+"
    r"|(?=\s+(?:because|therefore|hence|so that|and then|then|after|so)\b)",
    re.I,
)


@dataclass
class SpanStep:
    index: int
    text: str
    regions: list[str]
    occupancy: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "regions": list(self.regions),
            "occupancy": {k: round(v, 4) for k, v in self.occupancy.items() if v > 0},
        }


@dataclass
class RegionTrace:
    id: str
    does: str
    handles: str
    band: str
    series: list[float]
    peak_span: int | None
    peak_text: str
    first_on: int | None
    last_on: int | None
    variation: float
    delta: float
    varies: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "does": self.does,
            "handles": self.handles,
            "band": self.band,
            "series": [round(x, 4) for x in self.series],
            "peak_span": self.peak_span,
            "peak_text": self.peak_text,
            "first_on": self.first_on,
            "last_on": self.last_on,
            "variation": round(self.variation, 4),
            "delta": round(self.delta, 4),
            "varies": self.varies,
        }


@dataclass
class PromptTrace:
    text: str
    spans: list[SpanStep] = field(default_factory=list)
    regions: list[RegionTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
            "regions": [r.to_dict() for r in self.regions],
        }


def split_spans(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in _SPAN.split(raw) if p and p.strip()]
    return parts or [raw]


def _varies_sentence(rid: str, series: list[float], spans: list[str]) -> str:
    job = what_region_does(rid).split(" Band:")[0].rstrip(".")
    if not series or max(series) <= 0:
        return f"{rid} stays silent on this prompt. {job}."
    peak = int(max(range(len(series)), key=lambda i: series[i]))
    on = [i for i, v in enumerate(series) if v > 0]
    first, last = on[0], on[-1]
    peak_txt = spans[peak] if peak < len(spans) else ""
    if len(on) == 1:
        return (
            f"{rid} turns on only at span {peak} ({peak_txt!r}). {job}."
        )
    rising = series[-1] - series[0]
    trend = "rises toward motor" if rising > 0.05 else (
        "falls after the opening" if rising < -0.05 else "stays active across spans"
    )
    return (
        f"{rid} is on from span {first} to {last}, peaks at span {peak} "
        f"({peak_txt!r}), and {trend}. {job}."
    )


def trace_prompt(text: str) -> PromptTrace:
    """Compile each clause; return per-region series through the prompt."""
    spans = split_spans(text)
    steps: list[SpanStep] = []
    ids = region_ids()
    for i, span in enumerate(spans):
        plan = compile_regions(span)
        occ = plan.occupancy()
        steps.append(
            SpanStep(
                index=i,
                text=span,
                regions=list(plan.regions),
                occupancy=dict(occ),
            )
        )
    traces: list[RegionTrace] = []
    n = max(len(steps), 1)
    for rid in ids:
        series = [float(s.occupancy.get(rid, 0.0)) for s in steps] or [0.0]
        mean = sum(series) / n
        var = (sum((x - mean) ** 2 for x in series) / n) ** 0.5
        peak = int(max(range(len(series)), key=lambda i: series[i])) if series else 0
        on = [i for i, v in enumerate(series) if v > 0]
        r = REGION_BY_ID[rid]
        traces.append(
            RegionTrace(
                id=rid,
                does=what_region_does(rid),
                handles=r.handles,
                band=r.band,
                series=series,
                peak_span=peak if series and series[peak] > 0 else None,
                peak_text=spans[peak] if series and series[peak] > 0 and peak < len(spans) else "",
                first_on=on[0] if on else None,
                last_on=on[-1] if on else None,
                variation=var,
                delta=(series[-1] - series[0]) if series else 0.0,
                varies=_varies_sentence(rid, series, spans),
            )
        )
    return PromptTrace(text=text or "", spans=steps, regions=traces)
