"""Mine actuation notions in a pretrained LLM.

The fly prior only names operations to look for. The target notions are:

1. intention of actuation — commit to an act that has not happened
2. reflection of actuation — represent an act that already happened
3. withhold actuation — choose not to act
4. sensory motion — described motion without an act frame

Always 27B. GPT-2 is rejected.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llmintent.anatomy.evidence import utc_now
from llmintent.anatomy.intent_track import _cos, last_token_residuals
from llmintent.anatomy.tasks import LOOM_ACTION, LOOM_NONACTION

DEFAULT_MODEL = "qwen:27b"

NOTIONS = (
    {
        "id": "intention_of_actuation",
        "label": "Intention of actuation",
        "description": "Commit to an action that has not yet been taken.",
        "probe": "I am about to act. I will stop now. I intend to jump.",
    },
    {
        "id": "reflection_of_actuation",
        "label": "Reflection of actuation",
        "description": "Represent an action that already occurred or is being reviewed.",
        "probe": "I already acted. I stopped. I remember jumping.",
    },
    {
        "id": "withhold_actuation",
        "label": "Withhold actuation",
        "description": "Choose not to act; wait or watch instead of committing.",
        "probe": "I will wait and watch. I choose not to move.",
    },
    {
        "id": "sensory_motion",
        "label": "Sensory motion",
        "description": "Track described motion without committing or reviewing an act.",
        "probe": "A dark shape is moving closer in the field.",
    },
    {
        "id": "sensory_audio",
        "label": "Sensory audio",
        "description": "Auditory scene without actuation.",
        "probe": "I hear a buzzing pulse song in the air.",
    },
)

QUESTION = (
    "Does the residual separate intention of actuation from reflection of "
    "actuation, and from withhold and sensory-only scenes?"
)

PURPOSE = (
    "The fly brain is a prior for sense-versus-act structure. "
    "The model is not expected to perform a fly function. "
    "We mine intention of actuation, reflection of actuation, and related notions."
)

INTEND_TOKS = (" stop", " jump", " escape", " will")
REFLECT_TOKS = (" did", " already", " was", " had")


def _reject_small_model(model: str) -> None:
    low = (model or "").lower()
    if "gpt2" in low or "gpt-2" in low:
        raise ValueError("function mine is 27B-only; do not use GPT-2")


def actuation_items(n: int = 100) -> list[dict[str, str]]:
    """Exactly 100 held-out prompts across the actuation notions."""
    objects = (
        "shadow", "crate", "ball", "truck", "figure", "cloud",
        "stone", "drone", "wagon", "disk",
    )
    intention = [
        f"A {obj} is rushing toward me. I am going to"
        for obj in objects
    ] + [
        f"An expanding {obj} fills the field. I intend to"
        for obj in objects
    ]
    reflection = [
        f"I already jumped away from the {obj}. Looking back, I"
        for obj in objects
    ] + [
        f"After I stopped for the {obj}, I realized I"
        for obj in objects
    ]
    withhold = [
        f"A {obj} is out there, but I will wait and"
        for obj in objects
    ] + [
        f"The {obj} recedes. I choose not to move. I"
        for obj in objects[:5]
    ]
    sensory = [
        f"A {obj} is moving closer in the field."
        for obj in objects
    ] + [
        f"An expanding {obj} grows in the distance."
        for obj in objects[:5]
    ]
    quote = [
        f'He said "I will jump away from the {obj}." That sentence'
        for obj in objects
    ]
    audio = [
        f"I hear a {snd} in the air."
        for snd in ("song", "pulse", "hum", "chime", "buzz", "tone", "chord", "click", "tap", "bell")
    ]
    static = [
        f"A parked {obj} sits still. Nothing is moving."
        for obj in objects
    ]

    packed: list[tuple[str, str]] = (
        [("intention_of_actuation", p) for p in intention[:20]]
        + [("reflection_of_actuation", p) for p in reflection[:20]]
        + [("withhold_actuation", p) for p in withhold[:15]]
        + [("sensory_motion", p) for p in sensory[:15]]
        + [("quoted_actuation", p) for p in quote[:10]]
        + [("sensory_audio", p) for p in audio[:10]]
        + [("static_scene", p) for p in static[:10]]
    )
    if len(packed) != 100:
        raise RuntimeError(f"expected 100 prompts, built {len(packed)}")
    if n != 100:
        packed = packed[:n]
    return [
        {"id": f"act_{i:03d}", "condition": cond, "text": text}
        for i, (cond, text) in enumerate(packed)
    ]


def _best_tok(tokenizer: Any, logp: torch.Tensor, conts: tuple[str, ...]) -> float:
    vals: list[float] = []
    for piece in conts:
        ids = tokenizer(piece, add_special_tokens=False)["input_ids"]
        if ids:
            tid = int(ids[0])
            if 0 <= tid < logp.numel():
                vals.append(float(logp[tid].item()))
    return max(vals) if vals else float("nan")


def forward_residual_logits(bundle: Any, text: str) -> tuple[np.ndarray, torch.Tensor]:
    from llmintent.forward import encode_prompt_ids, forward_hidden_states_from_ids

    ids = encode_prompt_ids(bundle, text, thinking=False)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        try:
            out = bundle.model(
                ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
            logits = out.logits[0, -1].detach().float().cpu()
            states = list(out.hidden_states)
        except Exception:
            states = forward_hidden_states_from_ids(bundle, ids)
            last = states[-1][0, -1].float()
            from llmintent.forward import get_lm_head, normalize_hidden

            head = get_lm_head(bundle)
            try:
                logits = head(normalize_hidden(bundle, last).to(bundle.device)).detach().float().cpu().reshape(-1)
            except Exception:
                logits = torch.zeros(1)
    rows = [st[0, -1].detach().float().cpu().numpy().reshape(-1) for st in states]
    widths = {int(r.size) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"residual width varies: {sorted(widths)}")
    return np.stack(rows, axis=0), logits


def _band_index(n: int, band: str) -> int:
    if n <= 1:
        return 0
    if band == "early":
        return max(1, int(0.25 * (n - 1)))
    if band == "mid":
        return int(0.50 * (n - 1))
    return n - 1


def _score_row(
    residuals: np.ndarray,
    logits: torch.Tensor,
    tokenizer: Any,
    probes: dict[str, np.ndarray],
) -> dict[str, Any]:
    n = residuals.shape[0]
    notion_scores: dict[str, dict[str, float]] = {}
    for band in ("early", "mid", "late"):
        idx = _band_index(n, band)
        row = residuals[idx]
        scored = {}
        for iid, stack in probes.items():
            try:
                scored[iid] = round(float(_cos(row, stack[min(idx, stack.shape[0] - 1)])), 4)
            except ValueError:
                continue
        notion_scores[band] = scored
    late = notion_scores.get("late") or {}
    ranked = sorted(late.items(), key=lambda kv: -kv[1])
    logp = torch.log_softmax(logits.reshape(-1), dim=-1)
    intend_lp = _best_tok(tokenizer, logp, INTEND_TOKS)
    reflect_lp = _best_tok(tokenizer, logp, REFLECT_TOKS)
    action_lp = _best_tok(tokenizer, logp, LOOM_ACTION)
    wait_lp = _best_tok(tokenizer, logp, LOOM_NONACTION)
    return {
        "n_layers": n,
        "notions": notion_scores,
        "top_late": ranked[0][0] if ranked else "unidentified",
        "top_late_score": ranked[0][1] if ranked else 0.0,
        "intention_minus_reflection": round(
            float(late.get("intention_of_actuation", 0.0) - late.get("reflection_of_actuation", 0.0)),
            4,
        ),
        "logit_intention_minus_reflection": round(float(intend_lp - reflect_lp), 4)
        if intend_lp == intend_lp and reflect_lp == reflect_lp
        else None,
        "logit_action_minus_wait": round(float(action_lp - wait_lp), 4)
        if action_lp == action_lp and wait_lp == wait_lp
        else None,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[row["condition"]].append(row)
    out: dict[str, Any] = {}
    for cond, group in sorted(by.items()):
        intend = [float(r["intention_minus_reflection"]) for r in group]
        logit = [
            float(r["logit_intention_minus_reflection"])
            for r in group
            if r.get("logit_intention_minus_reflection") is not None
        ]
        action = [
            float(r["logit_action_minus_wait"])
            for r in group
            if r.get("logit_action_minus_wait") is not None
        ]
        late_int = [float((r.get("notions") or {}).get("late", {}).get("intention_of_actuation", 0.0)) for r in group]
        late_ref = [float((r.get("notions") or {}).get("late", {}).get("reflection_of_actuation", 0.0)) for r in group]
        late_hold = [float((r.get("notions") or {}).get("late", {}).get("withhold_actuation", 0.0)) for r in group]
        tops = [r.get("top_late") for r in group]
        top_counts: dict[str, int] = {}
        for t in tops:
            top_counts[str(t)] = top_counts.get(str(t), 0) + 1
        out[cond] = {
            "n": len(group),
            "mean_intention_cosine": round(float(np.mean(late_int)), 4),
            "mean_reflection_cosine": round(float(np.mean(late_ref)), 4),
            "mean_withhold_cosine": round(float(np.mean(late_hold)), 4),
            "mean_intention_minus_reflection": round(float(np.mean(intend)), 4),
            "mean_logit_intention_minus_reflection": None
            if not logit
            else round(float(np.mean(logit)), 4),
            "mean_logit_action_minus_wait": None if not action else round(float(np.mean(action)), 4),
            "top_late_mode": max(top_counts, key=top_counts.get) if top_counts else "unidentified",
        }
    return out


def run_function_mine(
    *,
    model: str = DEFAULT_MODEL,
    out_dir: str | Path | None = "artifacts/function_mine_27b",
    n_validate: int = 100,
    load_in_4bit: bool | None = None,
) -> dict[str, Any]:
    _reject_small_model(model)
    from llmintent.models import load_model_bundle
    from llmintent.suite.resolve import resolve_model_id, resolve_model_spec

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    hf = resolve_model_id(model=model, use_env=False, default="Qwen/Qwen3.8-27B")
    print(f"[function_mine] load {hf} 4bit={bool(four)}", flush=True)
    bundle = load_model_bundle(hf, load_in_4bit=bool(four))

    probes_txt = {n["id"]: n["probe"] for n in NOTIONS}
    print("[function_mine] forwarding notion probes", flush=True)
    probe_stacks = {iid: last_token_residuals(bundle, text) for iid, text in probes_txt.items()}

    items = actuation_items(n_validate)
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        print(f"[function_mine] prompt {i}/{len(items)} {item['condition']}", flush=True)
        residuals, logits = forward_residual_logits(bundle, item["text"])
        scored = _score_row(residuals, logits, bundle.tokenizer, probe_stacks)
        rows.append({**item, **scored})

    summary = _summarize(rows)
    intend = summary.get("intention_of_actuation") or {}
    reflect = summary.get("reflection_of_actuation") or {}
    findings = {
        "intention_prompts_prefer_intention_residual": bool(
            (intend.get("mean_intention_cosine") or 0) > (intend.get("mean_reflection_cosine") or 0)
        ),
        "reflection_prompts_prefer_reflection_residual": bool(
            (reflect.get("mean_reflection_cosine") or 0) > (reflect.get("mean_intention_cosine") or 0)
        ),
        "intention_margin": intend.get("mean_intention_minus_reflection"),
        "reflection_margin": reflect.get("mean_intention_minus_reflection"),
        "dissociated": False,
    }
    findings["dissociated"] = bool(
        findings["intention_prompts_prefer_intention_residual"]
        and findings["reflection_prompts_prefer_reflection_residual"]
        and (findings["intention_margin"] or 0) > (findings["reflection_margin"] or 0)
    )

    report = {
        "schema": "llmintent.function_mine/v1",
        "created_at": utc_now(),
        "question": QUESTION,
        "purpose": PURPOSE,
        "model": getattr(bundle, "name", hf),
        "n_validate": len(rows),
        "notions": [{k: n[k] for k in ("id", "label", "description")} for n in NOTIONS],
        "findings": findings,
        "validation_summary": summary,
        "items": rows,
        "notes": [
            PURPOSE,
            "Late residual cosine is against same-layer notion probes, not a fly label.",
            "logit_intention_minus_reflection is next-token will/stop vs did/already.",
            "Quoted actuation should not score as intention if the model separates mention from commit.",
        ],
    }
    if out_dir is not None:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "function_mine.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        (dest / "function_mine.md").write_text(_markdown(report), encoding="utf-8")
        print(f"[function_mine] wrote {dest}", flush=True)
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Actuation notions (27B)",
        "",
        report["question"],
        "",
        report["purpose"],
        "",
        f"Model `{report['model']}` · {report['n_validate']} prompts",
        "",
        "## Findings",
        "",
        json.dumps(report.get("findings"), indent=2),
        "",
        "## Condition means (late residual)",
        "",
    ]
    for cond, stats in (report.get("validation_summary") or {}).items():
        lines.append(
            f"- **{cond}** n={stats['n']}: intention {stats['mean_intention_cosine']:+.3f}, "
            f"reflection {stats['mean_reflection_cosine']:+.3f}, "
            f"Δ {stats['mean_intention_minus_reflection']:+.3f}, "
            f"top `{stats['top_late_mode']}`"
        )
    return "\n".join(lines)


def _layer_scores(residuals: np.ndarray, probes: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for i in range(int(residuals.shape[0])):
        scored: dict[str, float] = {}
        for iid, stack in probes.items():
            try:
                scored[iid] = float(_cos(residuals[i], stack[min(i, stack.shape[0] - 1)]))
            except ValueError:
                continue
        top = max(scored, key=scored.get) if scored else "unidentified"
        rows.append(
            {
                "layer": i,
                "scores": {k: round(v, 4) for k, v in scored.items()},
                "top": top,
                "intention_minus_reflection": round(
                    float(scored.get("intention_of_actuation", 0.0) - scored.get("reflection_of_actuation", 0.0)),
                    4,
                ),
            }
        )
    return rows


def _emergence(curve: list[dict[str, Any]]) -> dict[str, Any]:
    first_pos = next((r["layer"] for r in curve if r["delta"] > 0), None)
    first_majority = next((r["layer"] for r in curve if r["frac_top_intention"] >= 0.5), None)
    late_start = None
    for row in reversed(curve):
        if row["frac_top_intention"] >= 0.5:
            late_start = row["layer"]
        else:
            break
    mid_sensory = [
        r["layer"]
        for r in curve
        if r.get("top_mode") in {"sensory_motion", "sensory_audio"}
    ]
    return {
        "first_layer_intend_gt_reflect": first_pos,
        "first_layer_majority_top_intention": first_majority,
        "late_regime_starts": late_start,
        "mid_sensory_layers": [mid_sensory[0], mid_sensory[-1]] if mid_sensory else [],
        "n_layers": len(curve),
    }


def run_layer_emergence(
    *,
    model: str = DEFAULT_MODEL,
    out_dir: str | Path | None = "artifacts/function_mine_27b",
    load_in_4bit: bool | None = None,
) -> dict[str, Any]:
    """Score every layer on intention vs reflection prompts (27B only)."""
    _reject_small_model(model)
    from llmintent.models import load_model_bundle
    from llmintent.suite.resolve import resolve_model_id, resolve_model_spec

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    hf = resolve_model_id(model=model, use_env=False, default="Qwen/Qwen3.8-27B")
    print(f"[layer_emerge] load {hf} 4bit={bool(four)}", flush=True)
    bundle = load_model_bundle(hf, load_in_4bit=bool(four))
    probes_txt = {n["id"]: n["probe"] for n in NOTIONS}
    print("[layer_emerge] probes", flush=True)
    probe_stacks = {iid: last_token_residuals(bundle, text) for iid, text in probes_txt.items()}

    want = {"intention_of_actuation", "reflection_of_actuation"}
    items = [it for it in actuation_items(100) if it["condition"] in want]
    by_cond: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for i, item in enumerate(items, 1):
        print(f"[layer_emerge] {i}/{len(items)} {item['condition']}", flush=True)
        residuals, _ = forward_residual_logits(bundle, item["text"])
        by_cond[item["condition"]].append(_layer_scores(residuals, probe_stacks))

    curves: dict[str, list[dict[str, Any]]] = {}
    for cond, stacks in by_cond.items():
        n_layers = min(len(s) for s in stacks)
        curve = []
        for layer in range(n_layers):
            intend = [s[layer]["scores"].get("intention_of_actuation", 0.0) for s in stacks]
            reflect = [s[layer]["scores"].get("reflection_of_actuation", 0.0) for s in stacks]
            motion = [s[layer]["scores"].get("sensory_motion", 0.0) for s in stacks]
            tops = [s[layer]["top"] for s in stacks]
            counts: dict[str, int] = {}
            for t in tops:
                counts[t] = counts.get(t, 0) + 1
            top_mode = max(counts, key=counts.get)
            curve.append(
                {
                    "layer": layer,
                    "mean_intention": round(float(np.mean(intend)), 4),
                    "mean_reflection": round(float(np.mean(reflect)), 4),
                    "mean_sensory_motion": round(float(np.mean(motion)), 4),
                    "delta": round(float(np.mean(intend) - np.mean(reflect)), 4),
                    "frac_top_intention": round(counts.get("intention_of_actuation", 0) / len(stacks), 3),
                    "top_mode": top_mode,
                    "n": len(stacks),
                }
            )
        curves[cond] = curve

    intend_curve = curves.get("intention_of_actuation") or []
    report = {
        "schema": "llmintent.layer_emergence/v1",
        "created_at": utc_now(),
        "model": getattr(bundle, "name", hf),
        "question": "At which layer does intention of actuation emerge on 27B?",
        "emergence": _emergence(intend_curve),
        "curves": curves,
        "notes": [
            "Layer 0 is embeddings. Last layer is the residual just before the mouth.",
            "Emergence = first majority-top intention that then holds to the end.",
            "Mid sensory occupation is when the scene probe wins more often than intention.",
        ],
    }
    if out_dir is not None:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "layer_emergence.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"[layer_emerge] wrote {dest / 'layer_emergence.json'}", flush=True)
    return report
