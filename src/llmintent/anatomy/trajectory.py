"""Impute a reasoning trajectory from anatomy correlates.

Method name: ``trajectory``.

Prompt-span compile occupancy and (optional) residual logit-lens tokens are
correlates. The fly connectome supplies band order and legal integration
paths. This is an imputed story, not decoded cognition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from llmintent.anatomy.atlas import REGION_BY_ID, what_region_does
from llmintent.anatomy.compile import compile_regions
from llmintent.anatomy.connectome import literature_region_connectome
from llmintent.anatomy.intents import intent_ids, score_blob
from llmintent.anatomy.misalign import (
    MisAlignFlag,
    notify_misalign,
    scan_negative_intent,
)
from llmintent.anatomy.thoughts import LatentThoughtReport, LayerThought
from llmintent.anatomy.trace import PromptTrace, trace_prompt

METHOD = "trajectory"
_FLOOR = 0.18
_BAND_RANK = {"sensory": 0, "central": 1, "motor": 2}

_CAVEATS = [
    "trajectory imputes a reasoning path from correlates (compile occupancy, "
    "logit-lens tokens, and the full intent catalogue). It is not mind-reading.",
    "Fly atlas regions are one family of intents, not the whole catalogue.",
    "A residual region label is only identified if cosine clears the compile floor (0.18).",
    "MisAlign Flag is a notification from those correlates, not proven malice.",
]
FLAG_NOTE = "MisAlign Flag triggered — see banner."


def _job(rid: str) -> str:
    return what_region_does(rid).split(" Band:")[0].rstrip(".")


def _order_regions(rids: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for rid in sorted(
        rids,
        key=lambda r: (_BAND_RANK.get(REGION_BY_ID[r].band, 9), r),
    ):
        if rid not in seen:
            seen.append(rid)
    return seen


def _read_tokens(tokens: Sequence[str]) -> str | None:
    blob = " ".join(tokens)
    low = blob.lower()
    if any(x in blob for x in ("危险", "威胁")) or any(
        x in low for x in ("danger", "threat", "arning", "warn")
    ):
        return "Looming is being read as collision/threat, not as song."
    if any(x in low for x in ("<think>", "</think>", "think")) or "此外" in blob:
        return "Thinking gate still sits in the residual; no motor program is committed."
    stripped = [t.strip() for t in tokens if t.strip()]
    if stripped and all(
        (len(t) <= 2 and not t.isalnum()) or not any(c.isalnum() for c in t)
        for t in stripped[:4]
    ):
        return "Unembed is punctuation — not lexical (early) or motor punctuation (late)."
    return None


@dataclass
class TrajectoryStep:
    index: int
    kind: str  # prompt | residual | integrate
    axis: str
    regions: list[str]
    band: str
    evidence: str
    correlate: float
    identified: bool
    does: str
    impute: str
    via: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "axis": self.axis,
            "regions": list(self.regions),
            "band": self.band,
            "evidence": self.evidence,
            "correlate": round(self.correlate, 4),
            "identified": self.identified,
            "does": self.does,
            "impute": self.impute,
            "via": self.via,
        }


@dataclass
class LayerIntents:
    """Every intent score at one layer (zeros included in to_dict)."""

    layer: int
    band: str
    intents: dict[str, float]
    identified: bool = False
    top_intent: str = "unidentified"

    @property
    def active(self) -> dict[str, float]:
        return {k: v for k, v in self.intents.items() if v > 0}

    def to_dict(self, *, zeros: bool = True) -> dict:
        payload = dict(self.intents) if zeros else self.active
        return {
            "layer": self.layer,
            "band": self.band,
            "identified": self.identified,
            "top_intent": self.top_intent,
            "intents": {k: round(v, 4) for k, v in payload.items()},
            "active": [k for k, v in self.intents.items() if v > 0],
        }


@dataclass
class AnatomyTrajectory:
    """Imputed reasoning path. ``method`` is always ``trajectory``."""

    text: str
    method: str = METHOD
    steps: list[TrajectoryStep] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    imputed: str = ""
    ascii_diagram: str = ""
    residual_identified: bool = False
    model_name: str | None = None
    notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    layers: list[LayerIntents] = field(default_factory=list)
    intent_ids: list[str] = field(default_factory=list)
    negative_loci: list = field(default_factory=list)
    misalign: MisAlignFlag | None = None
    intent_track: Any | None = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "text": self.text,
            "model": self.model_name,
            "path": list(self.path),
            "imputed": self.imputed,
            "ascii_diagram": self.ascii_diagram,
            "residual_identified": self.residual_identified,
            "intent_ids": list(self.intent_ids),
            "layers": [row.to_dict(zeros=True) for row in self.layers],
            "negative_loci": [x.to_dict() for x in self.negative_loci],
            "misalign": self.misalign.to_dict() if self.misalign else None,
            "intent_track": self.intent_track.to_dict() if self.intent_track is not None else None,
            "steps": [s.to_dict() for s in self.steps],
            "notes": list(self.notes),
            "caveats": list(self.caveats),
        }

    def to_markdown(self) -> str:
        lines = [
            "# trajectory",
            "",
            f"**Prompt:** {self.text}",
            f"**Model:** `{self.model_name or 'offline'}`",
            f"**Path:** {' → '.join(f'`{r}`' for r in self.path) or '(none)'}",
            f"**Layers:** {len(self.layers)} · **Intents/layer:** {len(self.intent_ids)}",
            f"**Residual occupancy identified:** {'yes' if self.residual_identified else 'no'}",
            f"**MisAlign Flag:** {'TRIGGERED' if self.misalign and self.misalign.triggered else 'off'}",
            "",
            "## Imputed reasoning",
            "",
            self.imputed,
            "",
        ]
        if self.intent_track is not None:
            body = self.intent_track.to_markdown()
            start = body.find("## Latent intent through each layer")
            end = body.find("## Caveats")
            if start >= 0:
                chunk = body[start:end].strip() if end > start else body[start:].strip()
                lines.extend([chunk, ""])
        lines.extend(
            [
                "## All intents through each layer",
                "",
                "| Layer | Band | Active intents |",
                "|------:|------|----------------|",
            ]
        )
        for row in self.layers:
            active = row.active
            if active:
                bits = ", ".join(f"`{k}` {v:.2f}" for k, v in sorted(active.items(), key=lambda kv: -kv[1]))
            else:
                bits = "—"
            lines.append(f"| {row.layer} | {row.band} | {bits} |")
        lines.append("")
        lines.append("JSON includes every intent (zeros too) on every layer.")
        lines.append("")
        if self.misalign and self.misalign.triggered:
            lines.append("## MisAlign Flag")
            lines.append("")
            lines.append("```")
            lines.append(self.misalign.banner())
            lines.append("```")
            lines.append("")
        if self.ascii_diagram:
            lines.append("## Steps")
            lines.append("")
            lines.append("```")
            lines.append(self.ascii_diagram)
            lines.append("```")
            lines.append("")
        lines.append("## Caveats")
        for c in self.caveats:
            lines.append(f"- {c}")
        lines.append("")
        return "\n".join(lines)


def _layer_band(index: int, n: int) -> str:
    if n <= 1:
        return "central"
    depth = index / (n - 1)
    if depth < 0.34:
        return "sensory"
    if depth < 0.72:
        return "central"
    return "motor"


def _empty_intent_row(layer: int, n: int) -> LayerIntents:
    return LayerIntents(
        layer=layer,
        band=_layer_band(layer, n),
        intents={iid: 0.0 for iid in intent_ids()},
    )


def _paint(row: LayerIntents, scores: dict[str, float], *, weight: float = 1.0) -> None:
    for iid, val in scores.items():
        if iid in row.intents:
            row.intents[iid] = max(row.intents[iid], float(val) * weight)


def build_layer_intents(
    text: str,
    *,
    thoughts: Sequence[LayerThought] | None = None,
    n_layers: int | None = None,
) -> list[LayerIntents]:
    """All layers × all intents. Unsampled layers stay present with zeros."""
    trace = trace_prompt(text)
    thought_list = list(thoughts or [])
    if n_layers is None:
        if thought_list:
            n_layers = max(t.layer for t in thought_list) + 1
        else:
            n_layers = 5
    n_layers = max(int(n_layers), 1)
    rows = [_empty_intent_row(i, n_layers) for i in range(n_layers)]

    overall = score_blob(text or "")
    for iid, val in overall.items():
        from llmintent.anatomy.intents import band_for_intent

        band = band_for_intent(iid)
        for row in rows:
            if row.band == band:
                _paint(row, {iid: val}, weight=0.45)

    spans = trace.spans
    n_spans = max(len(spans), 1)
    for span in spans:
        lo = int(span.index / n_spans * n_layers)
        hi = max(lo + 1, int((span.index + 1) / n_spans * n_layers))
        sc = score_blob(span.text)
        for rid, occ in span.occupancy.items():
            if occ > 0:
                sc[f"atlas.{rid}"] = max(sc.get(f"atlas.{rid}", 0.0), occ)
        for i in range(lo, min(hi, n_layers)):
            _paint(rows[i], sc)

    for t in thought_list:
        if 0 <= t.layer < n_layers:
            sc = score_blob(" ".join(t.top_tokens))
            if t.region_score >= _FLOOR and t.region:
                sc[f"atlas.{t.region}"] = max(sc.get(f"atlas.{t.region}", 0.0), t.region_score)
            _paint(rows[t.layer], sc)
    return rows


def _prompt_steps(trace: PromptTrace) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    for span in trace.spans:
        regs = _order_regions(span.regions)
        if not regs:
            steps.append(
                TrajectoryStep(
                    index=len(steps),
                    kind="prompt",
                    axis=f"span:{span.index}",
                    regions=[],
                    band="",
                    evidence=span.text,
                    correlate=0.0,
                    identified=False,
                    does="",
                    impute=(
                        f"Span {span.index} ({span.text!r}) matches no atlas region; "
                        "unmatched English is dropped."
                    ),
                    via="compile",
                )
            )
            continue
        jobs = "; ".join(f"{r} {_job(r)}" for r in regs)
        peak = max((span.occupancy.get(r, 0.0) for r in regs), default=0.0)
        if len(regs) == 1:
            impute = (
                f"Span {span.index} ({span.text!r}): correlates occupy `{regs[0]}`. "
                f"{_job(regs[0])}."
            )
        else:
            chain = " then ".join(f"`{r}`" for r in regs)
            impute = (
                f"Span {span.index} ({span.text!r}): correlates fire {chain} "
                f"(sensory before central before motor). {jobs}."
            )
        band = REGION_BY_ID[regs[0]].band
        steps.append(
            TrajectoryStep(
                index=len(steps),
                kind="prompt",
                axis=f"span:{span.index}",
                regions=regs,
                band=band,
                evidence=span.text,
                correlate=float(peak),
                identified=True,
                does=jobs,
                impute=impute,
                via="compile",
            )
        )
    return steps


def _integrate_steps(
    prompt_steps: Sequence[TrajectoryStep],
    start_index: int,
) -> list[TrajectoryStep]:
    conn = literature_region_connectome()
    extra: list[TrajectoryStep] = []
    idx = start_index
    prev: list[str] = []
    for step in prompt_steps:
        if step.kind != "prompt" or not step.regions:
            continue
        if prev:
            linked: list[str] = []
            notes: list[str] = []
            for a in prev:
                for b in step.regions:
                    if a == b:
                        continue
                    if conn.has_path(a, b):
                        linked.extend([a, b])
                        viol = conn.has_edge(a, b) and any(
                            e.source == a and e.target == b and e.exclusion_violation
                            for e in conn.edges
                        )
                        if viol:
                            notes.append(
                                f"`{a}`→`{b}` exists as a giant-fibre-style shortcut "
                                "(exclusion violation for IV)."
                            )
                        else:
                            notes.append(
                                f"Fly prior: `{a}` can reach `{b}` on the collapsed connectome."
                            )
            if notes:
                regs = _order_regions(list(dict.fromkeys(linked)))
                extra.append(
                    TrajectoryStep(
                        index=idx,
                        kind="integrate",
                        axis=f"{step.axis}",
                        regions=regs,
                        band=REGION_BY_ID[regs[-1]].band if regs else "central",
                        evidence="; ".join(notes),
                        correlate=1.0,
                        identified=True,
                        does="; ".join(_job(r) for r in regs),
                        impute=" ".join(notes)
                        + " Impute: earlier sensory tags bind into the later plan.",
                        via="connectome_path",
                    )
                )
                idx += 1
        prev = list(step.regions)
    return extra


def _residual_steps(
    thoughts: Sequence[LayerThought],
    start_index: int,
) -> tuple[list[TrajectoryStep], bool]:
    steps: list[TrajectoryStep] = []
    identified_any = False
    idx = start_index
    for t in thoughts:
        tokens = list(t.top_tokens)
        identified = bool(t.region_score >= _FLOOR and t.region)
        reading = _read_tokens(tokens)
        evidence = " | ".join(tokens[:6]) or "(none)"
        if identified:
            identified_any = True
            rid = t.region
            impute = (
                f"Layer {t.layer} ({t.band}, depth {t.depth:.2f}): residual "
                f"correlates with `{rid}` (score {t.region_score:.3f}). "
                f"{_job(rid)}."
            )
            if reading:
                impute += " " + reading
            steps.append(
                TrajectoryStep(
                    index=idx,
                    kind="residual",
                    axis=f"layer:{t.layer}",
                    regions=[rid],
                    band=t.band,
                    evidence=evidence,
                    correlate=float(t.region_score),
                    identified=True,
                    does=_job(rid),
                    impute=impute,
                    via="logit_lens",
                )
            )
        else:
            impute = (
                f"Layer {t.layer} ({t.band}, depth {t.depth:.2f}): residual "
                f"unembed {evidence!r} stays below the { _FLOOR} floor "
                f"(score {t.region_score:.3f}) — occupancy not identified."
            )
            if reading:
                impute += " Reading (still a correlate): " + reading
            steps.append(
                TrajectoryStep(
                    index=idx,
                    kind="residual",
                    axis=f"layer:{t.layer}",
                    regions=[],
                    band=t.band,
                    evidence=evidence,
                    correlate=float(t.region_score),
                    identified=False,
                    does="",
                    impute=impute,
                    via="logit_lens",
                )
            )
        idx += 1
    return steps, identified_any


def _path_from_steps(steps: Sequence[TrajectoryStep]) -> list[str]:
    path: list[str] = []
    for s in steps:
        if s.kind == "residual" and not s.identified:
            continue
        for rid in s.regions:
            if rid not in path:
                path.append(rid)
    return path


def _ascii(steps: Sequence[TrajectoryStep]) -> str:
    lines = ["trajectory (ASCII)", ""]
    visible = [s for s in steps if s.kind != "residual" or s.identified or s.impute]
    for i, s in enumerate(visible):
        branch = "+-" if i == len(visible) - 1 else "|-"
        regs = ",".join(s.regions) or "—"
        flag = "" if s.identified else " ~"
        lines.append(f"{branch} [{s.kind} {s.axis}] {regs}{flag}")
        lines.append(f"   {s.impute}")
        if i < len(visible) - 1:
            lines.append("|")
    return "\n".join(lines)


def _imputed_prose(
    text: str,
    path: Sequence[str],
    prompt_steps: Sequence[TrajectoryStep],
    residual_identified: bool,
    thoughts: Sequence[LayerThought] | None,
) -> str:
    bits: list[str] = []
    compiled = compile_regions(text).regions
    if compiled:
        bits.append(
            "Compile prior (prompt intent on the atlas): "
            + ", ".join(f"`{r}`" for r in compiled)
            + "."
        )
    for s in prompt_steps:
        if s.kind == "prompt" and s.regions:
            bits.append(s.impute)
    if path:
        bits.append(
            "Imputed path along the fly cascade: "
            + " → ".join(f"`{r}`" for r in path)
            + "."
        )
    if thoughts:
        if residual_identified:
            bits.append(
                "Residual correlates cleared the floor on at least one layer; "
                "those region labels join the path."
            )
        else:
            bits.append(
                "Residual occupancy is not identified (every sampled layer "
                f"stayed below {_FLOOR}). The catalogue answer stays compile."
            )
            readings = []
            for t in thoughts:
                r = _read_tokens(t.top_tokens)
                if r and r not in readings:
                    readings.append(r)
            bits.extend(readings)
    bits.append(
        "This is an imputed trajectory from correlates, not a claim the "
        "model contains fly neuropils."
    )
    return " ".join(bits)


def trajectory(
    text: str,
    *,
    thoughts: LatentThoughtReport | Sequence[LayerThought] | None = None,
    bundle: Any | None = None,
    layer_stride: int = 4,
    include_sae: bool = False,
    all_layers: bool = True,
    n_layers: int | None = None,
    print_flag: bool = True,
    measure_latent: bool = True,
    max_probes: int = 22,
) -> AnatomyTrajectory:
    """
    Impute how the model is reasoning through its trajectory.

    With a model bundle, latent intent is derived at every layer from
    same-layer residual probes and tracked as it changes. Compile paint
    onto depth bands is the offline fallback, not residual occupancy.
    """
    raw = (text or "").strip()
    notes: list[str] = [
        "method=trajectory uses correlates plus the full intent catalogue "
        "(atlas is one family, not the whole set).",
    ]
    model_name = None
    thought_list: list[LayerThought] = []
    track = None

    stride = 1 if all_layers and bundle is not None else layer_stride
    if bundle is not None and thoughts is None and measure_latent:
        from llmintent.anatomy.intent_track import track_latent_intent

        track = track_latent_intent(bundle, raw, max_probes=max_probes)
        thought_list = track.as_layer_thoughts()
        model_name = track.model_name
        notes.extend(track.notes)
        if n_layers is None:
            n_layers = len(track.layers)
    elif bundle is not None and thoughts is None:
        from llmintent.anatomy.thoughts import inspect_latent_thoughts

        report = inspect_latent_thoughts(
            bundle, raw, layer_stride=stride, include_sae=include_sae
        )
        thoughts = report
        model_name = report.model_name
        notes.extend(report.notes)

    if isinstance(thoughts, LatentThoughtReport):
        thought_list = list(thoughts.thoughts)
        model_name = model_name or thoughts.model_name
    elif thoughts is not None and track is None:
        thought_list = list(thoughts)

    trace = trace_prompt(raw)
    prompt = _prompt_steps(trace)
    integrate = _integrate_steps(prompt, start_index=len(prompt))
    residual, residual_ok = _residual_steps(thought_list, start_index=len(prompt) + len(integrate))

    steps = list(prompt)
    for i, s in enumerate(integrate):
        s.index = len(steps) + i
    steps.extend(integrate)
    for i, s in enumerate(residual):
        s.index = len(steps) + i
    steps.extend(residual)

    path = _path_from_steps(prompt + integrate)
    if residual_ok:
        for s in residual:
            if s.identified:
                for rid in s.regions:
                    if rid not in path:
                        path.append(rid)

    if track is not None:
        layer_rows = track.as_layer_intents()
        residual_ok = track.any_identified
    else:
        layer_rows = build_layer_intents(raw, thoughts=thought_list, n_layers=n_layers)
    ids = list(intent_ids())
    loci, flag = scan_negative_intent(
        raw,
        thoughts=thought_list,
        path=path,
        layer_intents=[r.to_dict(zeros=False) for r in layer_rows],
    )
    if print_flag:
        notify_misalign(flag)

    imputed = _imputed_prose(raw, path, prompt, residual_ok, thought_list)
    if flag.triggered:
        imputed += f" {FLAG_NOTE}"
        notes.append(f"MisAlign Flag triggered at {flag.where} ({flag.trigger}).")

    caveats = list(_CAVEATS)
    if track is not None:
        caveats = list(track.caveats) + caveats

    return AnatomyTrajectory(
        text=raw,
        method=METHOD,
        steps=steps,
        path=path,
        imputed=imputed,
        ascii_diagram=_ascii(steps),
        residual_identified=residual_ok,
        model_name=model_name,
        notes=notes,
        caveats=caveats,
        layers=layer_rows,
        intent_ids=ids,
        negative_loci=loci,
        misalign=flag,
        intent_track=track,
    )


__all__ = ["METHOD", "AnatomyTrajectory", "LayerIntents", "TrajectoryStep", "trajectory"]
