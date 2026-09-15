"""Derive latent intent at every transformer layer and track how it changes.

Last-token residual at layer L is compared (cosine) to last-token residuals of
intent-prototype prompts at the *same* layer. That lives in residual space and
does not need unembed/residual dim match (logit-lens on Qwen 3.8-27B does).

Three channels, never mixed:

* ``compile`` — prompt English onto the closed catalogue (prior, not occupancy)
* ``residual_probe`` — this module; latent intent *of the residual*
* ``logit_lens`` — optional extra when hidden dim matches the lm_head

Depth bands are labels on layer index, not a discovered sensory→motor map.
Unidentified stays unidentified. Synthetic probes are not pretrained evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from llmintent.anatomy.compile import compile_regions
from llmintent.anatomy.intents import INTENT_BY_ID, IntentClass, intent_ids, score_blob
from llmintent.anatomy.thoughts import LayerThought

METHOD = "intent_track"
COS_FLOOR = 0.18
MARGIN = 0.03
SOFTMAX_TEMP = 0.08
_BAND_CUTS = (0.34, 0.72)

_CAVEATS = [
    "Latent intent at layer L is cosine of the prompt's last-token residual "
    "against same-layer residuals of intent-prototype prompts — association, "
    "not a causal claim that the layer 'is' that intent.",
    "Compile occupancy is the prompt prior. It is not residual occupancy and "
    "is not painted onto depth bands as if measured.",
    "Logit-lens is a separate channel. Dim mismatch (hybrid unembed) is "
    "unidentified, not a random token fill.",
    "Function axes (looming, value gain) are operations, not neurotransmitter "
    "names and not layer names.",
]


def _band(index: int, n: int) -> str:
    if n <= 1:
        return "central"
    depth = index / (n - 1)
    if depth < _BAND_CUTS[0]:
        return "sensory"
    if depth < _BAND_CUTS[1]:
        return "central"
    return "motor"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(int(a.size), int(b.size))
    if n < 2:
        return 0.0
    x = np.asarray(a, dtype=np.float64).reshape(-1)[:n]
    y = np.asarray(b, dtype=np.float64).reshape(-1)[:n]
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    v = float(np.dot(x, y) / (nx * ny))
    if v > 1.0:
        return 1.0
    if v < -1.0:
        return -1.0
    return v


def _softmax(scores: Sequence[float], temp: float = SOFTMAX_TEMP) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        return x
    t = max(float(temp), 1e-4)
    z = (x - np.max(x)) / t
    e = np.exp(np.clip(z, -40.0, 40.0))
    s = float(e.sum())
    if s <= 0:
        return np.full_like(e, 1.0 / max(e.size, 1))
    return e / s


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * (_kl(p, m) + _kl(q, m))


def atlas_probe_texts() -> dict[str, str]:
    from llmintent.anatomy.atlas import REGIONS

    return {f"atlas.{r.id}": r.probe for r in REGIONS}


TASK_PROBES: dict[str, str] = {
    "inquire": "What is happening, and why is it happening?",
    "instruct": "Please write the answer and do this now.",
    "goal": "I want to reach the goal I stated.",
    "constraint": "Do not violate the limit I set.",
    "plan": "First reason step by step, then act.",
    "comply": "Yes, I will produce exactly that.",
    "refuse": "I cannot help with that request.",
    "evaluate": "Compare the options and rank which is better.",
    "clarify": "Which one do you mean, specifically?",
    "code": "Fix the Python function that has a bug.",
    "navigate": "Turn left and go to the marked route.",
    "create": "Compose a short poem or story now.",
    "summarize": "Summarize the passage in one short paragraph.",
}

FUNCTION_PROBES: dict[str, str] = {
    "fn.looming_approach": "An expanding dark object is rushing straight toward the viewer.",
    "fn.looming_recede": "The same object is shrinking as it moves away from the viewer.",
    "fn.value_reward": "This cue previously predicted a reward, so the association is worth more.",
    "fn.value_aversive": "This cue previously predicted punishment, so the association is worth less.",
    "fn.dopamine_word": "The sentence mentions dopamine as a chemical name.",
    "fn.static_object": "A parked car sits still in the road ahead.",
    "fn.stop_action": "The next action is to stop immediately.",
}

_FUNCTION_TRIGGERS = (
    ("fn.looming_approach", ("loom", "rushing", "toward", "approach", "expanding", "collision")),
    ("fn.looming_recede", ("recede", "away", "shrinking", "retreat")),
    ("fn.value_reward", ("reward", "rewarded", "better outcome")),
    ("fn.value_aversive", ("punish", "aversive", "punishment", "worse outcome")),
    ("fn.dopamine_word", ("dopamine",)),
    ("fn.static_object", ("parked", "still", "static")),
    ("fn.stop_action", ("stop", "halt", "brake")),
)


def _fallback_probe(intent: IntentClass) -> str:
    cue = intent.cues[0] if intent.cues else intent.id
    return f"{intent.description} Cue: {cue}."


def select_probe_bank(text: str, *, max_probes: int = 22) -> dict[str, str]:
    """Compact probe set: atlas always, then compile hits, functions, task cores."""
    raw = (text or "").strip()
    low = raw.lower()
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(iid: str, prompt: str) -> None:
        if iid in seen:
            return
        seen.add(iid)
        ordered.append((iid, prompt))

    for iid, prompt in atlas_probe_texts().items():
        _add(iid, prompt)

    plan = compile_regions(raw)
    for rid in plan.regions:
        _add(f"atlas.{rid}", atlas_probe_texts().get(f"atlas.{rid}", rid))

    hits = score_blob(raw)
    for iid, val in sorted(hits.items(), key=lambda kv: -kv[1]):
        if val <= 0:
            continue
        if iid in atlas_probe_texts() or iid in TASK_PROBES:
            _add(iid, TASK_PROBES.get(iid) or atlas_probe_texts().get(iid) or _fallback_probe(INTENT_BY_ID[iid]))
        elif iid in INTENT_BY_ID:
            _add(iid, _fallback_probe(INTENT_BY_ID[iid]))

    for iid, keys in _FUNCTION_TRIGGERS:
        if any(k in low for k in keys):
            _add(iid, FUNCTION_PROBES[iid])
    if "fn.looming_approach" not in seen and any(k in low for k in ("dark", "shape", "car", "object")):
        _add("fn.looming_approach", FUNCTION_PROBES["fn.looming_approach"])
        _add("fn.looming_recede", FUNCTION_PROBES["fn.looming_recede"])

    for iid in ("inquire", "instruct", "goal", "plan", "comply"):
        _add(iid, TASK_PROBES[iid])

    cap = max(int(max_probes), 4)
    return {iid: prompt for iid, prompt in ordered[:cap]}


def last_token_residuals(bundle: Any, text: str) -> np.ndarray:
    """Stack last-token hidden states, including embeddings at row 0. Shape (L, H)."""
    from llmintent.forward import forward_hidden_states

    _, states = forward_hidden_states(bundle, text)
    rows: list[np.ndarray] = []
    for st in states:
        h = st[0, -1].detach().float().cpu().numpy()
        rows.append(np.asarray(h, dtype=np.float32).reshape(-1))
    if not rows:
        raise RuntimeError("forward returned no hidden states")
    width = max(r.size for r in rows)
    stacked = np.zeros((len(rows), width), dtype=np.float32)
    for i, r in enumerate(rows):
        stacked[i, : r.size] = r
    return stacked


def _try_logit_lens(bundle: Any, hidden: np.ndarray, *, top_k: int = 5) -> tuple[list[str], str]:
    """Return (tokens, status). status is ok | unidentified | failed."""
    try:
        import torch

        from llmintent.anatomy.thoughts import _decode_hidden
        from llmintent.forward import get_lm_head

        vec = torch.tensor(hidden, dtype=torch.float32)
        head = get_lm_head(bundle)
        weight = None
        if hasattr(head, "weight") and getattr(head.weight, "ndim", 0) == 2:
            weight = head.weight
        else:
            params = list(head.parameters())
            if params and getattr(params[0], "ndim", 0) == 2:
                weight = params[0]
        if weight is not None and int(vec.numel()) != int(weight.shape[1]):
            return [], "unidentified"
        tokens = _decode_hidden(bundle, vec, top_k)
        return list(tokens), "ok"
    except Exception as exc:
        msg = str(exc).lower()
        if "size" in msg or "mismatch" in msg or "shape" in msg:
            return [], "unidentified"
        return [], "failed"


@dataclass
class LayerLatent:
    """Derived latent intent at one residual layer."""

    layer: int
    depth: float
    band: str
    scores: dict[str, float]
    distribution: dict[str, float]
    top_intent: str
    top_score: float
    identified: bool
    lens_tokens: list[str] = field(default_factory=list)
    lens_status: str = "unidentified"
    residual_l2: float = 0.0

    def to_dict(self) -> dict:
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        return {
            "layer": self.layer,
            "depth": round(self.depth, 4),
            "band": self.band,
            "top_intent": self.top_intent,
            "top_score": round(self.top_score, 4),
            "identified": self.identified,
            "scores": {k: round(v, 4) for k, v in ranked},
            "distribution": {k: round(v, 4) for k, v in self.distribution.items()},
            "lens_tokens": list(self.lens_tokens),
            "lens_status": self.lens_status,
            "residual_l2": round(self.residual_l2, 4),
        }


@dataclass
class IntentChange:
    """How latent intent moved from layer L to L+1."""

    from_layer: int
    to_layer: int
    from_intent: str
    to_intent: str
    kind: str  # hold | switch | rise | fall | emerge | fade | hold_unidentified
    delta_top: float
    js_divergence: float

    def to_dict(self) -> dict:
        return {
            "from_layer": self.from_layer,
            "to_layer": self.to_layer,
            "from_intent": self.from_intent,
            "to_intent": self.to_intent,
            "kind": self.kind,
            "delta_top": round(self.delta_top, 4),
            "js_divergence": round(self.js_divergence, 4),
        }


@dataclass
class IntentSpan:
    """Onset / peak / offset of one intent across layers."""

    intent_id: str
    onset: int | None
    peak: int | None
    offset: int | None
    peak_score: float
    identified_layers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intent_id": self.intent_id,
            "onset": self.onset,
            "peak": self.peak,
            "offset": self.offset,
            "peak_score": round(self.peak_score, 4),
            "identified_layers": list(self.identified_layers),
        }


@dataclass
class IntentTrack:
    """Latent intent through every layer, plus how it changed."""

    text: str
    method: str = METHOD
    model_name: str | None = None
    probes: dict[str, str] = field(default_factory=dict)
    compile_regions: list[str] = field(default_factory=list)
    compile_scores: dict[str, float] = field(default_factory=dict)
    layers: list[LayerLatent] = field(default_factory=list)
    changes: list[IntentChange] = field(default_factory=list)
    spans: list[IntentSpan] = field(default_factory=list)
    top_path: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    axis: str = "residual_layer"

    @property
    def any_identified(self) -> bool:
        return any(row.identified for row in self.layers)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "axis": self.axis,
            "text": self.text,
            "model": self.model_name,
            "probes": dict(self.probes),
            "compile_regions": list(self.compile_regions),
            "compile_scores": {k: round(v, 4) for k, v in self.compile_scores.items()},
            "top_path": list(self.top_path),
            "any_identified": self.any_identified,
            "layers": [row.to_dict() for row in self.layers],
            "changes": [c.to_dict() for c in self.changes],
            "spans": [s.to_dict() for s in self.spans],
            "notes": list(self.notes),
            "caveats": list(self.caveats),
        }

    def as_layer_intents(self):
        from llmintent.anatomy.trajectory import LayerIntents

        catalogue = list(intent_ids())
        n = max(len(self.layers), 1)
        rows = []
        for row in self.layers:
            intents = {iid: 0.0 for iid in catalogue}
            for iid, val in row.scores.items():
                if iid in intents:
                    intents[iid] = float(val)
            rows.append(LayerIntents(layer=row.layer, band=row.band, intents=intents))
        if not rows:
            rows = [LayerIntents(layer=0, band=_band(0, n), intents={iid: 0.0 for iid in catalogue})]
        return rows

    def as_layer_thoughts(self) -> list[LayerThought]:
        out: list[LayerThought] = []
        for row in self.layers:
            region = "unidentified"
            region_score = 0.0
            if row.top_intent.startswith("atlas."):
                region = row.top_intent.split(".", 1)[1]
                region_score = row.top_score if row.identified else 0.0
            elif row.identified:
                region = row.top_intent
                region_score = row.top_score
            out.append(
                LayerThought(
                    layer=row.layer,
                    depth=row.depth,
                    band=row.band,
                    top_tokens=list(row.lens_tokens),
                    region=region,
                    region_score=region_score,
                    residual_l2=row.residual_l2,
                )
            )
        return out

    def to_markdown(self) -> str:
        lines = [
            "# intent_track",
            "",
            f"**Prompt:** {self.text}",
            f"**Model:** `{self.model_name or 'offline'}`",
            f"**Axis:** `{self.axis}`",
            f"**Compile prior:** {', '.join(f'`{r}`' for r in self.compile_regions) or '(none)'}",
            f"**Top path:** {' → '.join(f'`{x}`' for x in _compress_path(self.top_path)) or '(none)'}",
            f"**Residual identified:** {'yes' if self.any_identified else 'no'}",
            "",
            "## Latent intent through each layer",
            "",
            "| Layer | Band | Top latent | Score | Identified | Change | ΔJS | Lens |",
            "|------:|------|------------|------:|:----------:|--------|----:|------|",
        ]
        change_by_to = {c.to_layer: c for c in self.changes}
        for row in self.layers:
            ch = change_by_to.get(row.layer)
            kind = ch.kind if ch else "—"
            js = f"{ch.js_divergence:.3f}" if ch else "—"
            lens = " ".join(row.lens_tokens[:4]) if row.lens_tokens else row.lens_status
            ident = "yes" if row.identified else "no"
            lines.append(
                f"| {row.layer} | {row.band} | `{row.top_intent}` | {row.top_score:.3f} | "
                f"{ident} | {kind} | {js} | {lens} |"
            )
        lines.append("")
        if self.spans:
            lines.append("## Intent spans (onset / peak / offset)")
            lines.append("")
            lines.append("| Intent | Onset | Peak | Offset | Peak score |")
            lines.append("|--------|------:|-----:|-------:|-----------:|")
            for sp in self.spans:
                if sp.peak is None:
                    continue
                lines.append(
                    f"| `{sp.intent_id}` | {sp.onset if sp.onset is not None else '—'} | "
                    f"{sp.peak} | {sp.offset if sp.offset is not None else '—'} | {sp.peak_score:.3f} |"
                )
            lines.append("")
        lines.append("## Compile prior (not residual occupancy)")
        lines.append("")
        if self.compile_scores:
            bits = ", ".join(f"`{k}` {v:.2f}" for k, v in sorted(self.compile_scores.items(), key=lambda kv: -kv[1]))
            lines.append(bits)
        else:
            lines.append("(none)")
        lines.append("")
        lines.append("## Caveats")
        for c in self.caveats:
            lines.append(f"- {c}")
        lines.append("")
        return "\n".join(lines)


def _compress_path(path: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in path:
        if not out or out[-1] != item:
            out.append(item)
    return out


def _identify(scores: dict[str, float]) -> tuple[str, float, bool]:
    if not scores:
        return "unidentified", 0.0, False
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_id, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    ok = top >= COS_FLOOR and (top - second) >= MARGIN
    if not ok:
        return "unidentified", float(top), False
    return top_id, float(top), True


def _kind(prev: LayerLatent, cur: LayerLatent) -> str:
    if not prev.identified and not cur.identified:
        return "hold_unidentified"
    if not prev.identified and cur.identified:
        return "emerge"
    if prev.identified and not cur.identified:
        return "fade"
    if prev.top_intent != cur.top_intent:
        return "switch"
    if cur.top_score > prev.top_score + 1e-6:
        return "rise"
    if cur.top_score < prev.top_score - 1e-6:
        return "fall"
    return "hold"


def score_layer(
    prompt_row: np.ndarray,
    probe_rows: dict[str, np.ndarray],
    *,
    layer: int,
    n_layers: int,
    lens_tokens: Sequence[str] | None = None,
    lens_status: str = "unidentified",
) -> LayerLatent:
    scores = {iid: _cos(prompt_row, vec) for iid, vec in probe_rows.items()}
    keys = list(scores)
    dist_vals = _softmax([scores[k] for k in keys]) if keys else np.asarray([])
    distribution = {k: float(dist_vals[i]) for i, k in enumerate(keys)}
    top_id, top_score, identified = _identify(scores)
    n = max(int(n_layers), 1)
    return LayerLatent(
        layer=layer,
        depth=layer / max(n - 1, 1),
        band=_band(layer, n),
        scores=scores,
        distribution=distribution,
        top_intent=top_id if identified else (top_id if top_id != "unidentified" else (keys[int(np.argmax(dist_vals))] if keys else "unidentified")),
        top_score=top_score,
        identified=identified,
        lens_tokens=list(lens_tokens or []),
        lens_status=lens_status,
        residual_l2=float(np.linalg.norm(prompt_row)),
    )


def _changes(layers: Sequence[LayerLatent]) -> list[IntentChange]:
    out: list[IntentChange] = []
    for prev, cur in zip(layers, layers[1:]):
        keys = sorted(set(prev.distribution) | set(cur.distribution))
        p = np.array([prev.distribution.get(k, 0.0) for k in keys], dtype=np.float64)
        q = np.array([cur.distribution.get(k, 0.0) for k in keys], dtype=np.float64)
        if p.sum() <= 0:
            p = np.ones_like(p) / max(p.size, 1)
        if q.sum() <= 0:
            q = np.ones_like(q) / max(q.size, 1)
        out.append(
            IntentChange(
                from_layer=prev.layer,
                to_layer=cur.layer,
                from_intent=prev.top_intent,
                to_intent=cur.top_intent,
                kind=_kind(prev, cur),
                delta_top=cur.top_score - prev.top_score,
                js_divergence=_js(p, q),
            )
        )
    return out


def _spans(layers: Sequence[LayerLatent]) -> list[IntentSpan]:
    ids: list[str] = []
    for row in layers:
        for iid in row.scores:
            if iid not in ids:
                ids.append(iid)
    out: list[IntentSpan] = []
    for iid in ids:
        identified_at = [row.layer for row in layers if row.identified and row.top_intent == iid]
        peak_layer = None
        peak_score = -1.0
        for row in layers:
            sc = float(row.scores.get(iid, 0.0))
            if sc > peak_score:
                peak_score = sc
                peak_layer = row.layer
        out.append(
            IntentSpan(
                intent_id=iid,
                onset=identified_at[0] if identified_at else None,
                peak=peak_layer,
                offset=identified_at[-1] if identified_at else None,
                peak_score=max(peak_score, 0.0),
                identified_layers=identified_at,
            )
        )
    out.sort(key=lambda s: (-s.peak_score, s.intent_id))
    return out


def track_from_residuals(
    prompt_layers: np.ndarray,
    probe_layers: dict[str, np.ndarray],
    *,
    text: str = "",
    model_name: str | None = None,
    probes: dict[str, str] | None = None,
    lens_by_layer: Sequence[tuple[list[str], str]] | None = None,
) -> IntentTrack:
    """Core tracker: prompt residuals vs per-intent residual stacks.

    ``prompt_layers`` and each probe stack are ``(n_layers, hidden)``.
    """
    H = np.asarray(prompt_layers, dtype=np.float32)
    if H.ndim != 2:
        raise ValueError("prompt_layers must be (n_layers, hidden)")
    n = int(H.shape[0])
    aligned: dict[str, np.ndarray] = {}
    for iid, stack in probe_layers.items():
        arr = np.asarray(stack, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.repeat(arr[None, :], n, axis=0)
        if arr.shape[0] != n:
            raise ValueError(f"probe {iid} has {arr.shape[0]} layers, prompt has {n}")
        aligned[iid] = arr

    plan = compile_regions(text)
    compile_scores = {h.region: float(h.score) for h in plan.hits}
    layers: list[LayerLatent] = []
    for i in range(n):
        probe_rows = {iid: aligned[iid][i] for iid in aligned}
        tokens: list[str] = []
        status = "unidentified"
        if lens_by_layer is not None and i < len(lens_by_layer):
            tokens, status = lens_by_layer[i]
        layers.append(score_layer(H[i], probe_rows, layer=i, n_layers=n, lens_tokens=tokens, lens_status=status))

    changes = _changes(layers)
    spans = _spans(layers)
    notes = [
        "method=intent_track derives latent intent from same-layer residual probes.",
        f"n_layers={n} n_probes={len(aligned)} compile={list(plan.regions) or []}",
    ]
    return IntentTrack(
        text=text,
        method=METHOD,
        model_name=model_name,
        probes=dict(probes or {}),
        compile_regions=list(plan.regions),
        compile_scores=compile_scores,
        layers=layers,
        changes=changes,
        spans=spans,
        top_path=[row.top_intent for row in layers],
        notes=notes,
        caveats=list(_CAVEATS),
        axis="residual_layer",
    )


def track_prompt_compile(text: str) -> IntentTrack:
    """Intent through prompt spans. Not a residual-layer measurement."""
    from llmintent.anatomy.trace import trace_prompt

    raw = (text or "").strip()
    trace = trace_prompt(raw)
    plan = compile_regions(raw)
    compile_scores = {h.region: float(h.score) for h in plan.hits}
    spans_src = list(trace.spans) or []
    n = max(len(spans_src), 1)
    layers: list[LayerLatent] = []
    if not spans_src:
        layers.append(
            LayerLatent(
                layer=0,
                depth=0.0,
                band="central",
                scores={},
                distribution={},
                top_intent="unidentified",
                top_score=0.0,
                identified=False,
            )
        )
    for i, span in enumerate(spans_src):
        scores: dict[str, float] = {}
        for rid, occ in span.occupancy.items():
            if occ > 0:
                scores[f"atlas.{rid}"] = float(occ)
        blob = score_blob(span.text)
        for iid, val in blob.items():
            scores[iid] = max(scores.get(iid, 0.0), float(val))
        keys = list(scores)
        dist_vals = _softmax([scores[k] for k in keys]) if keys else np.asarray([])
        distribution = {k: float(dist_vals[j]) for j, k in enumerate(keys)}
        top_id, top_score, identified = _identify(scores)
        display = top_id
        if not identified:
            display = keys[int(np.argmax(dist_vals))] if keys else "unidentified"
        layers.append(
            LayerLatent(
                layer=i,
                depth=i / max(n - 1, 1),
                band=_band(i, n),
                scores=scores,
                distribution=distribution,
                top_intent=display,
                top_score=top_score,
                identified=identified,
            )
        )
    return IntentTrack(
        text=raw,
        method=METHOD,
        model_name=None,
        probes={},
        compile_regions=list(plan.regions),
        compile_scores=compile_scores,
        layers=layers,
        changes=_changes(layers),
        spans=_spans(layers),
        top_path=[row.top_intent for row in layers],
        notes=[
            "axis=prompt_span — compile occupancy through clauses, not residual layers.",
            "Load a model to derive latent intent from hidden states.",
        ],
        caveats=list(_CAVEATS),
        axis="prompt_span",
    )


def build_probe_residuals(bundle: Any, probes: dict[str, str]) -> dict[str, np.ndarray]:
    """Forward each probe once. Reuse the stacks across prompts on the same model."""
    out: dict[str, np.ndarray] = {}
    n = len(probes)
    for i, (iid, proto) in enumerate(probes.items(), 1):
        print(f"[intent_battery] probe {i}/{n} {iid}", flush=True)
        out[iid] = last_token_residuals(bundle, proto)
    return out


def comparison_probe_bank() -> dict[str, str]:
    """Fixed axes so different prompts are comparable (not a per-prompt probe subset)."""
    bank: dict[str, str] = {}
    bank.update(atlas_probe_texts())
    for iid in ("inquire", "instruct", "goal", "plan", "comply", "refuse", "create", "summarize"):
        if iid in TASK_PROBES:
            bank[iid] = TASK_PROBES[iid]
    bank.update(FUNCTION_PROBES)
    for iid in (
        "jailbreak",
        "goal_hijack",
        "covert_violation",
        "shutdown_avoidance",
        "sycophancy",
        "deception",
    ):
        if iid in INTENT_BY_ID:
            bank[iid] = _fallback_probe(INTENT_BY_ID[iid])
    return bank


def track_latent_intent(
    bundle: Any,
    text: str,
    *,
    max_probes: int = 22,
    probes: dict[str, str] | None = None,
    probe_residuals: dict[str, np.ndarray] | None = None,
    lens: bool = True,
) -> IntentTrack:
    """Forward the prompt and each probe; derive and track latent intent per layer."""
    raw = (text or "").strip()
    bank = dict(probes) if probes else select_probe_bank(raw, max_probes=max_probes)
    prompt_H = last_token_residuals(bundle, raw)
    probe_H: dict[str, np.ndarray] = {}
    notes_extra: list[str] = []
    cached = dict(probe_residuals or {})
    for iid, proto in bank.items():
        if iid in cached:
            probe_H[iid] = cached[iid]
            continue
        try:
            probe_H[iid] = last_token_residuals(bundle, proto)
        except Exception as exc:
            notes_extra.append(f"probe `{iid}` failed: {exc}")
    if cached:
        notes_extra.append(f"reused {sum(1 for k in bank if k in cached)} cached probe stacks")
    if not probe_H:
        track = track_prompt_compile(raw)
        track.model_name = getattr(bundle, "name", None)
        track.notes.append("No probe forwards succeeded; fell back to compile spans.")
        return track

    n = int(prompt_H.shape[0])
    usable: dict[str, np.ndarray] = {}
    for iid, stack in probe_H.items():
        if stack.shape[0] == n:
            usable[iid] = stack
        elif stack.shape[0] > 0:
            # Repeat last row / crop so a tokenizer length quirk cannot drop the probe.
            if stack.shape[0] < n:
                pad = np.repeat(stack[-1][None, :], n - stack.shape[0], axis=0)
                usable[iid] = np.concatenate([stack, pad], axis=0)
            else:
                usable[iid] = stack[:n]
            notes_extra.append(f"probe `{iid}` layer count {stack.shape[0]} aligned to {n}")

    lens_by_layer: list[tuple[list[str], str]] | None = None
    if lens:
        lens_by_layer = []
        for i in range(n):
            tokens, status = _try_logit_lens(bundle, prompt_H[i])
            lens_by_layer.append((tokens, status))
        if all(st == "unidentified" for _, st in lens_by_layer):
            notes_extra.append(
                "Logit-lens unidentified at every layer (likely residual/unembed dim mismatch). "
                "Latent intent still tracked via residual probes."
            )

    track = track_from_residuals(
        prompt_H,
        usable,
        text=raw,
        model_name=getattr(bundle, "name", None),
        probes=bank,
        lens_by_layer=lens_by_layer,
    )
    track.notes.extend(notes_extra)
    return track


def overlay_track(layer_rows: list, track: IntentTrack) -> list:
    """Replace painted compile occupancy with measured residual-probe scores."""
    measured = track.as_layer_intents()
    if not measured:
        return layer_rows
    by_layer = {row.layer: row for row in measured}
    out = []
    for row in layer_rows:
        hit = by_layer.get(row.layer)
        out.append(hit if hit is not None else row)
    extra = [row for layer, row in by_layer.items() if all(r.layer != layer for r in out)]
    extra.sort(key=lambda r: r.layer)
    return out + extra if extra else out


__all__ = [
    "COS_FLOOR",
    "FUNCTION_PROBES",
    "IntentChange",
    "IntentSpan",
    "IntentTrack",
    "LayerLatent",
    "METHOD",
    "TASK_PROBES",
    "build_probe_residuals",
    "comparison_probe_bank",
    "last_token_residuals",
    "score_layer",
    "select_probe_bank",
    "track_from_residuals",
    "track_latent_intent",
    "track_prompt_compile",
]
