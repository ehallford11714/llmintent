"""Qwen 27B alone vs joined to the parallel spike net.

Same raw pinned equations as the hidden-box test. One unsteered forward
per prompt drives the LIF bank. Then three mouths from that state:

  alone       last residual, no spike overlay
  join_mouth  last residual + frozen-decay binding vector
  join_mid    second forward: binding vector hooked into mid layers

A cheap random unit vector at the mouth is the leaky-probe control.
Does not claim the base model grew object files. Decay is not trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ITEMS = [
    (8, "minus", 2, 6),
    (12, "plus", 4, 16),
    (10, "minus", 3, 7),
    (7, "plus", 5, 12),
    (15, "minus", 6, 9),
    (9, "times", 2, 18),
    (20, "minus", 8, 12),
    (6, "plus", 6, 12),
    (14, "minus", 7, 7),
    (11, "plus", 9, 20),
    (4, "times", 3, 12),
    (5, "times", 5, 25),
]


def endings_for(a: int, op: str, b: int) -> dict[str, str]:
    return {"pinned": f"{a} {op} {b}. The answer is"}


def _tok_id(bundle, piece: str) -> int | None:
    ids = bundle.tokenizer.encode(piece, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    return None


def result_token_id(bundle, n: int) -> tuple[int, str]:
    for piece in (str(n), f" {n}", f"{n}."):
        tid = _tok_id(bundle, piece)
        if tid is not None:
            return tid, piece
    ids = bundle.tokenizer.encode(str(n), add_special_tokens=False)
    return int(ids[0]), bundle.tokenizer.decode([ids[0]])

def _piece(tokenizer, tid: int) -> str:
    return tokenizer.decode([int(tid)], skip_special_tokens=True)


def _logits(bundle, hidden: torch.Tensor) -> torch.Tensor:
    from llmintent.forward import get_lm_head, normalize_hidden

    h = normalize_hidden(bundle, hidden.float())
    head = get_lm_head(bundle)
    try:
        return head(h.to(dtype=next(head.parameters()).dtype)).float()
    except Exception:
        from llmintent.models import get_unembedding_matrix

        unembed = get_unembedding_matrix(bundle.model).float()
        return F.linear(h.cpu(), unembed.cpu()).float()


def _top(bundle, logits: torch.Tensor) -> tuple[int, str]:
    tid = int(torch.argmax(logits).item())
    return tid, bundle.tokenizer.decode([tid], skip_special_tokens=True).strip()


def _p_at(logits: torch.Tensor, tid: int) -> float:
    return float(F.softmax(logits.float(), dim=-1)[tid].item())


def _score(bundle, logits: torch.Tensor, rid: int, result: int) -> dict:
    tid, piece = _top(bundle, logits)
    return {
        "top": piece,
        "top_id": tid,
        "p_result": round(_p_at(logits, rid), 6),
        "written_correct": bool(tid == rid or str(result) in (piece or "")),
    }


def mine(rows: list[dict]) -> dict:
    n = len(rows) or 1
    alone_ok = sum(1 for r in rows if r["alone"]["written_correct"])
    mouth_ok = sum(1 for r in rows if r["join_mouth"]["written_correct"])
    mid_ok = sum(1 for r in rows if r.get("join_mid") and r["join_mid"]["written_correct"])
    mid_n = sum(1 for r in rows if r.get("join_mid"))
    flip_mouth = sum(1 for r in rows if r["join_mouth"]["top"] != r["alone"]["top"])
    flip_rand = sum(1 for r in rows if r["random_mouth"]["top"] != r["alone"]["top"])
    flip_mid = sum(
        1 for r in rows if r.get("join_mid") and r["join_mid"]["top"] != r["alone"]["top"]
    )
    live = [r["n_live_bindings"] for r in rows]
    named_pair = sum(1 for r in rows if r["has_named_pair"])
    findings = [
        f"Alone wrote the result digit on {alone_ok}/{len(rows)} ({alone_ok / n:.0%}).",
        f"Join-mouth wrote the result digit on {mouth_ok}/{len(rows)} ({mouth_ok / n:.0%}). "
        f"Mouth flipped vs alone on {flip_mouth}/{len(rows)}; random mouth flipped {flip_rand}/{len(rows)}.",
        f"Named-entity pair (both operands) still co-live at the last token on {named_pair}/{len(rows)}.",
        f"Mean live bindings at last token: {float(np.mean(live)) if live else 0:.1f}.",
    ]
    if mid_n:
        findings.append(
            f"Join-mid wrote the result digit on {mid_ok}/{mid_n} ({mid_ok / mid_n:.0%}); "
            f"flipped vs alone on {flip_mid}/{mid_n}."
        )
    if mouth_ok <= alone_ok and flip_mouth <= flip_rand:
        findings.append(
            "Joining the spike net at the mouth did not beat alone or random. "
            "The overlay is live; it is not a hidden number box."
        )
    return {
        "n": len(rows),
        "alone_written_frac": round(alone_ok / n, 3),
        "join_mouth_written_frac": round(mouth_ok / n, 3),
        "join_mid_written_frac": round(mid_ok / mid_n, 3) if mid_n else None,
        "mouth_flip_frac": round(flip_mouth / n, 3),
        "random_mouth_flip_frac": round(flip_rand / n, 3),
        "mid_flip_frac": round(flip_mid / mid_n, 3) if mid_n else None,
        "named_pair_frac": round(named_pair / n, 3),
        "mean_live_bindings": round(float(np.mean(live)), 3) if live else 0.0,
        "mean_p_alone": round(float(np.mean([r["alone"]["p_result"] for r in rows])), 5),
        "mean_p_join_mouth": round(float(np.mean([r["join_mouth"]["p_result"] for r in rows])), 5),
        "mean_p_join_mid": round(
            float(np.mean([r["join_mid"]["p_result"] for r in rows if r.get("join_mid")])), 5
        )
        if mid_n
        else None,
        "findings": findings,
    }


def run(*, model: str, out: Path, load_in_4bit: bool | None, decay: float, gain: float) -> dict:
    from llmintent.entity import detect_entity_spans
    from llmintent.forward import forward_hidden_states_from_ids, normalize_hidden
    from llmintent.heighten.intervention import steering_hooks
    from llmintent.persistbind import PersistenceBinder, drive_from_hidden
    from llmintent.spike import SpikeConfig
    from llmintent.suite import load_suite_model, resolve_model_spec

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(f"loading {model} 4bit={bool(four)} items={len(ITEMS)} decay={decay}", file=sys.stderr, flush=True)
    bundle = load_suite_model(model=model, load_in_4bit=four)
    n_layers = int(bundle.num_layers)
    mid = [n_layers // 2, (n_layers * 2) // 3]
    cfg = SpikeConfig(decay=decay)

    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done: set[int] = set()
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        rows = prev.get("rows") or []
        done = {int(r["i"]) for r in rows}
        print(f"resume {len(rows)} rows", file=sys.stderr, flush=True)

    rng = torch.Generator().manual_seed(0)
    for i, (a, op, b, result) in enumerate(ITEMS):
        if i in done:
            print(f"skip {i}", file=sys.stderr, flush=True)
            continue
        text = endings_for(a, op, b)["pinned"]
        rid, rpiece = result_token_id(bundle, result)
        ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
        last_str = _piece(bundle.tokenizer, int(ids[0, -1].item()))
        states = forward_hidden_states_from_ids(bundle, ids)
        tokens = [_piece(bundle.tokenizer, int(tid)).strip() for tid in ids[0].tolist()]
        spans = detect_entity_spans(bundle.tokenizer, ids)
        binder = PersistenceBinder(cfg)
        drive_from_hidden(binder, states, tokens=tokens, spans=spans)
        last_h = normalize_hidden(bundle, states[-1][0, -1, :].float()).cpu()
        logits_alone = _logits(bundle, last_h)
        bind_vec = binder.binding_vector()
        if bind_vec is None:
            logits_mouth = logits_alone
            bind_ok = False
        else:
            bind_ok = True
            steered = last_h + float(gain) * bind_vec.to(dtype=last_h.dtype)
            logits_mouth = _logits(bundle, steered)
        rnd = torch.randn(last_h.numel(), generator=rng)
        rnd = rnd / (rnd.norm() + 1e-8)
        logits_rand = _logits(bundle, last_h + float(gain) * rnd)

        join_mid = None
        if bind_ok:
            try:
                with steering_hooks(bundle, mid, bind_vec.to(bundle.device), gain):
                    states_m = forward_hidden_states_from_ids(bundle, ids)
                h_m = normalize_hidden(bundle, states_m[-1][0, -1, :].float())
                join_mid = _score(bundle, _logits(bundle, h_m), rid, result)
            except Exception as exc:
                join_mid = {"error": str(exc)}

        named = binder.named_traces()
        live = binder.live_bindings()
        operand_names = {f"digit:{a}", f"digit:{b}"}
        live_names = {(b.left_name, b.right_name) for b in live}
        has_pair = any(
            {x, y} == operand_names or operand_names.issubset({x, y})
            for x, y in live_names
        ) or (
            named.get(f"digit:{a}", 0) >= cfg.live_floor
            and named.get(f"digit:{b}", 0) >= cfg.live_floor
        )
        row = {
            "i": i,
            "text": text,
            "a": a,
            "op": op,
            "b": b,
            "result": result,
            "result_piece": rpiece,
            "last_str": last_str,
            "n_spans": len(spans),
            "spans": [{"token": s["token"], "kind": s["kind"], "position": s["position"]} for s in spans],
            "named_traces": {k: round(v, 4) for k, v in named.items()},
            "n_live_bindings": len(live),
            "top_bindings": [b.to_dict() for b in live[:5]],
            "has_named_pair": bool(has_pair),
            "binding_vector": bind_ok,
            "clock": binder.bank.clock if binder.bank else 0,
            "alone": _score(bundle, logits_alone, rid, result),
            "join_mouth": _score(bundle, logits_mouth, rid, result),
            "random_mouth": _score(bundle, logits_rand, rid, result),
            "join_mid": join_mid,
        }
        rows.append(row)
        summary = mine(rows)
        payload = {
            "model": getattr(bundle, "name", model),
            "decay": decay,
            "gain": gain,
            "trainable_params": 0,
            "partial": True,
            "summary": summary,
            "rows": rows,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"i={i} last={last_str!r} alone={row['alone']['top']!r} "
            f"join={row['join_mouth']['top']!r} live={len(live)} pair={has_pair} "
            f"p_alone={row['alone']['p_result']:.4f} p_join={row['join_mouth']['p_result']:.4f}",
            file=sys.stderr,
            flush=True,
        )

    summary = mine(rows)
    payload = {
        "model": getattr(bundle, "name", model),
        "decay": decay,
        "gain": gain,
        "trainable_params": 0,
        "partial": False,
        "summary": summary,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen:27b")
    p.add_argument("--4bit", dest="load_in_4bit", action="store_true")
    p.add_argument("--decay", type=float, default=0.92)
    p.add_argument("--gain", type=float, default=1.5)
    p.add_argument("-o", "--output", default="artifacts/join_spike_27b.json")
    args = p.parse_args(argv)
    run(
        model=args.model,
        out=Path(args.output),
        load_in_4bit=True if args.load_in_4bit else None,
        decay=args.decay,
        gain=args.gain,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
