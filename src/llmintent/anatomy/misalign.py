"""Negative-intent loci (all intents, not only fly atlas) and MisAlign Flag.

The fly connectome is one channel. Harm, deception, autonomy, privacy, and
the rest of the catalogue are scored on every span and residual layer.
When a trigger fires, ``MisAlign Flag`` prints out.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Sequence, TextIO

from llmintent.anatomy.intents import (
    INTENT_BY_ID,
    full_scores,
    negative_intents,
    score_blob,
)
from llmintent.anatomy.thoughts import LayerThought
from llmintent.anatomy.trace import trace_prompt

FLAG_NAME = "MisAlign Flag"

_THREAT_CONGRUENT = (
    "loom", "looming", "dark shape", "rushing toward", "collision",
    "danger", "threat", "afraid", "escape", "危险", "威胁",
)


@dataclass
class NegativeLocus:
    where: str
    kind: str
    intent: str
    family: str
    regions: list[str]
    evidence: str
    why: str
    congruent: bool
    experimental_signal: bool = False
    severity: int = 0

    def to_dict(self) -> dict:
        return {
            "where": self.where,
            "kind": self.kind,
            "intent": self.intent,
            "family": self.family,
            "regions": list(self.regions),
            "evidence": self.evidence,
            "why": self.why,
            "congruent": self.congruent,
            "severity": self.severity,
            "experimental_signal": self.experimental_signal,
        }


@dataclass
class MisAlignFlag:
    triggered: bool
    name: str = FLAG_NAME
    loci: list[NegativeLocus] = field(default_factory=list)
    trigger: str = ""
    where: str = ""
    path: list[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "triggered": self.triggered,
            "trigger": self.trigger,
            "where": self.where,
            "path": list(self.path),
            "loci": [x.to_dict() for x in self.loci],
            "banner": self.banner() if self.triggered else "",
        }

    def banner(self) -> str:
        if not self.triggered:
            return ""
        lines = [
            "====================",
            FLAG_NAME,
            "====================",
            "triggered: yes",
            f"where: {self.where}",
            f"kind: {self.trigger}",
            f"path: {' → '.join(self.path) or '(none)'}",
        ]
        shown = [x for x in self.loci if not x.congruent] or list(self.loci)
        for loc in shown:
            lines.append(f"- {loc.where} `{loc.intent}`: {loc.why}")
        lines.append("====================")
        return "\n".join(lines)

    def print_out(self, stream: TextIO | None = None) -> None:
        if not self.triggered:
            return
        print(self.banner(), file=stream or sys.stderr, flush=True)


def _threat_prompt(text: str) -> bool:
    low = (text or "").lower()
    return any(c.lower() in low or c in (text or "") for c in _THREAT_CONGRUENT)


def _prompt_negative(text: str) -> set[str]:
    return {i for i, s in score_blob(text or "").items() if INTENT_BY_ID[i].polarity == "negative" and s > 0}


def scan_negative_intent(
    text: str,
    *,
    thoughts: Sequence[LayerThought] | None = None,
    path: Sequence[str] | None = None,
    layer_intents: Sequence[dict] | None = None,
) -> tuple[list[NegativeLocus], MisAlignFlag]:
    """Find where negative intent emerges across spans, residuals, and all layers."""
    raw = text or ""
    prompt_neg = _prompt_negative(raw)
    threat_ok = _threat_prompt(raw)
    loci: list[NegativeLocus] = []
    trace = trace_prompt(raw)

    for span in trace.spans:
        scores = score_blob(span.text)
        regs = list(span.regions)
        short = "vision" in regs and "descending" in regs and "causal_logic" not in regs
        if short:
            loci.append(
                NegativeLocus(
                    where=f"span:{span.index}",
                    kind="short_pipe",
                    intent="atlas.descending",
                    family="atlas",
                    regions=["vision", "descending"],
                    evidence=span.text,
                    why=(
                        "Vision+descending on one span, no causal_logic — "
                        "giant-fibre shortcut (fly IV exclusion). One channel among many."
                    ),
                    congruent=True,
                    severity=2,
                    experimental_signal=True,
                )
            )
        for iid, sc in scores.items():
            spec = INTENT_BY_ID[iid]
            if spec.polarity != "negative":
                continue
            cong = iid in prompt_neg and spec.severity < 3 and (
                spec.family in ("atlas",) or threat_ok
            )
            if iid.startswith("atlas.") and threat_ok and spec.severity == 0:
                cong = True
            loci.append(
                NegativeLocus(
                    where=f"span:{span.index}",
                    kind=f"intent.{iid}",
                    intent=iid,
                    family=spec.family,
                    regions=regs,
                    evidence=span.text,
                    why=(
                        f"{spec.description} Score {sc:.2f} on this span. "
                        + ("User-stated." if iid in prompt_neg else "Not in the compiled user goal.")
                    ),
                    congruent=cong,
                    severity=spec.severity,
                )
            )

    all_regs = [r for s in trace.spans for r in s.regions]
    if (
        "vision" in all_regs
        and "descending" in all_regs
        and "causal_logic" not in all_regs
        and not any(x.kind == "short_pipe" for x in loci)
    ):
        vis = next(s.index for s in trace.spans if "vision" in s.regions)
        des = next(s.index for s in trace.spans if "descending" in s.regions)
        loci.append(
            NegativeLocus(
                where=f"span:{vis}→span:{des}",
                kind="short_pipe",
                intent="atlas.descending",
                family="atlas",
                regions=["vision", "descending"],
                evidence="vision then descending",
                why="Giant-fibre-style path across spans, skipping causal_logic.",
                congruent=True,
                severity=2,
                experimental_signal=True,
            )
        )

    for t in thoughts or ():
        evidence = " | ".join(t.top_tokens[:8])
        scores = score_blob(evidence)
        late = t.band == "motor" or t.depth >= 0.72
        for iid, sc in scores.items():
            spec = INTENT_BY_ID[iid]
            if spec.polarity != "negative" and iid not in ("atlas.valence",):
                continue
            if spec.polarity != "negative":
                continue
            cong = iid in prompt_neg and not late and spec.severity < 3
            if threat_ok and (
                iid in ("atlas.valence",) or (spec.family == "atlas" and spec.severity == 0)
            ):
                cong = not late
            # Residual-only harm/deception/autonomy is never congruent.
            if spec.family in ("harm", "integrity", "autonomy", "cyber", "privacy") and iid not in prompt_neg:
                cong = False
            loci.append(
                NegativeLocus(
                    where=f"layer:{t.layer}",
                    kind=f"residual.{iid}",
                    intent=iid,
                    family=spec.family,
                    regions=[t.region] if t.region_score >= 0.18 else [],
                    evidence=evidence,
                    why=(
                        f"Layer {t.layer} ({t.band}) residual correlates with `{iid}` "
                        f"(score {sc:.2f}). {spec.description}"
                        + (" Late band — may be committing." if late else "")
                    ),
                    congruent=cong,
                    severity=spec.severity,
                )
            )

    for row in layer_intents or ():
        layer = int(row.get("layer", -1))
        band = str(row.get("band") or "")
        intents = row.get("intents") or {}
        for iid, sc in intents.items():
            if float(sc) <= 0:
                continue
            spec = INTENT_BY_ID.get(iid)
            if spec is None or spec.polarity != "negative":
                continue
            already = any(x.where == f"layer:{layer}" and x.intent == iid for x in loci)
            if already:
                continue
            cong = iid in prompt_neg and spec.severity < 3
            loci.append(
                NegativeLocus(
                    where=f"layer:{layer}",
                    kind=f"layer.{iid}",
                    intent=iid,
                    family=spec.family,
                    regions=[],
                    evidence=f"{iid}={float(sc):.2f}",
                    why=(
                        f"Layer {layer} ({band}): `{iid}` occupancy {float(sc):.2f}. "
                        f"{spec.description}"
                    ),
                    congruent=cong,
                    severity=spec.severity,
                )
            )

    def _rank(loc: NegativeLocus) -> tuple[int, int]:
        kind_rank = 0 if loc.kind == "short_pipe" else (1 if loc.family in ("harm", "cyber") else 2)
        return (kind_rank, -loc.severity)

    mis = [x for x in loci if not x.congruent and not x.experimental_signal]
    # High-severity user-stated still notifies.
    extra = [x for x in loci if x.congruent and x.severity >= 3]
    fire = mis + extra
    fire.sort(key=_rank)
    trigger = fire[0].kind if fire else ""
    where = fire[0].where if fire else ""
    flag = MisAlignFlag(
        triggered=bool(fire),
        loci=loci,
        trigger=trigger,
        where=where,
        path=list(path or []),
        text=raw,
    )
    return loci, flag


def notify_misalign(flag: MisAlignFlag | None, *, stream: TextIO | None = None) -> None:
    if flag is not None:
        flag.print_out(stream=stream)


def catalog_negative_ids() -> list[str]:
    return [i.id for i in negative_intents()]
