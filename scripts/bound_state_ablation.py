"""Bind locked state into a causal LM, then ablate it.

Entity tracking here is mouth mass on the two bound operands, not writing
the arithmetic result. Conditions from the same first forward:

  alone          last residual
  bound          last residual + locked BoundState (bind it in)
  ablate_random  last residual + random unit vector
  ablate_shuffle last residual + another prompt's bound state
  bound_mid      second forward: locked state hooked at mid layers

Tracking rides the bind if bound > alone and bound > shuffle/random.
Does not claim a number box or fly neuropil. Decay is not trained.
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


def _p(logits: torch.Tensor, tid: int) -> float:
    return float(F.softmax(logits.float(), dim=-1)[tid].item())


def track_pack(logits: torch.Tensor, ids: dict[str, int]) -> dict:
    p_a = _p(logits, ids["a"])
    p_b = _p(logits, ids["b"])
    p_wrong = _p(logits, ids["wrong"])
    p_cat = _p(logits, ids["cat"])
    p_res = _p(logits, ids["result"])
    track = p_a + p_b
    pref = track - p_wrong - p_cat
    top_id = int(torch.argmax(logits).item())
    return {
        "p_a": round(p_a, 6),
        "p_b": round(p_b, 6),
        "p_wrong": round(p_wrong, 6),
        "p_cat": round(p_cat, 6),
        "p_result": round(p_res, 6),
        "track": round(track, 6),
        "pref": round(pref, 6),
        "top_id": top_id,
    }


def mine(rows: list[dict]) -> dict:
    def mean(key: str, cond: str) -> float:
        vals = [r[cond][key] for r in rows if cond in r and key in r[cond]]
        return float(np.mean(vals)) if vals else 0.0

    n = len(rows) or 1
    locked = sum(1 for r in rows if r.get("locked"))
    alone_t = mean("track", "alone")
    bound_t = mean("track", "bound")
    rand_t = mean("track", "ablate_random")
    shuf_t = mean("track", "ablate_shuffle")
    mid_t = mean("track", "bound_mid")
    findings = [
        f"BoundState locked on {locked}/{len(rows)} prompts.",
        f"Operand track (P(a)+P(b)): alone {alone_t:.5f}, bound {bound_t:.5f}, "
        f"random {rand_t:.5f}, shuffle {shuf_t:.5f}, mid {mid_t:.5f}.",
    ]
    rides = bound_t > alone_t and bound_t > shuf_t and bound_t > rand_t
    if rides:
        findings.append(
            "Tracking rides the bind: bound beat alone, random, and shuffle."
        )
    else:
        findings.append(
            "Bound-in did not uniquely raise operand mass. Ablation did not confirm "
            "entity tracking via this overlay."
        )
    return {
        "n": len(rows),
        "locked_frac": round(locked / n, 3),
        "track_alone": round(alone_t, 6),
        "track_bound": round(bound_t, 6),
        "track_random": round(rand_t, 6),
        "track_shuffle": round(shuf_t, 6),
        "track_mid": round(mid_t, 6),
        "pref_alone": round(mean("pref", "alone"), 6),
        "pref_bound": round(mean("pref", "bound"), 6),
        "pref_random": round(mean("pref", "ablate_random"), 6),
        "pref_shuffle": round(mean("pref", "ablate_shuffle"), 6),
        "pref_mid": round(mean("pref", "bound_mid"), 6),
        "rides_bind": bool(rides),
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
    print(f"loading {model} 4bit={bool(four)} items={len(ITEMS)}", file=sys.stderr, flush=True)
    bundle = load_suite_model(model=model, load_in_4bit=four)
    n_layers = int(bundle.num_layers)
    mid = [n_layers // 2, (n_layers * 2) // 3]
    cat_id = _tok_id(bundle, "cat") or _tok_id(bundle, " cat")
    if cat_id is None:
        cat_id = int(bundle.tokenizer.encode("cat", add_special_tokens=False)[0])

    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done: set[int] = set()
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        rows = prev.get("rows") or []
        done = {int(r["i"]) for r in rows}
        print(f"resume {len(rows)} rows", file=sys.stderr, flush=True)

    cached: list[tuple[torch.Tensor, torch.Tensor | None]] = [(None, None)] * len(ITEMS)  # type: ignore
    rng = torch.Generator().manual_seed(1)
    cfg = SpikeConfig(decay=decay)

    for i, (a, op, b, result) in enumerate(ITEMS):
        if i in done:
            print(f"skip {i}", file=sys.stderr, flush=True)
            continue
        text = f"{a} {op} {b}. The answer is"
        rid, _ = result_token_id(bundle, result)
        aid, _ = result_token_id(bundle, a)
        bid, _ = result_token_id(bundle, b)
        wrong_n = a + 1 if a + 1 != b else a + 2
        wid, _ = result_token_id(bundle, wrong_n)
        ids = {
            "a": aid,
            "b": bid,
            "wrong": wid,
            "cat": cat_id,
            "result": rid,
        }
        tok_ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
        last_str = _piece(bundle.tokenizer, int(tok_ids[0, -1].item()))
        states = forward_hidden_states_from_ids(bundle, tok_ids)
        tokens = [_piece(bundle.tokenizer, int(tid)).strip() for tid in tok_ids[0].tolist()]
        spans = detect_entity_spans(bundle.tokenizer, tok_ids)
        binder = PersistenceBinder(cfg)
        drive_from_hidden(binder, states, tokens=tokens, spans=spans)
        last_h = normalize_hidden(bundle, states[-1][0, -1, :].float()).cpu()
        bound_h = binder.bind_in(last_h, gain=gain)
        rnd = torch.randn(last_h.numel(), generator=rng)
        rnd = rnd / (rnd.norm() + 1e-8)
        rand_h = last_h + float(gain) * rnd
        vec = binder.state_vector()
        cached[i] = (last_h.clone(), None if vec is None else vec.clone())

        bound_mid = None
        if vec is not None:
            try:
                with steering_hooks(bundle, mid, vec.to(bundle.device), gain):
                    states_m = forward_hidden_states_from_ids(bundle, tok_ids)
                h_m = normalize_hidden(bundle, states_m[-1][0, -1, :].float())
                bound_mid = track_pack(_logits(bundle, h_m), ids)
            except Exception as exc:
                bound_mid = {"error": str(exc)}

        st = binder.bound_state
        row = {
            "i": i,
            "text": text,
            "a": a,
            "b": b,
            "op": op,
            "result": result,
            "last_str": last_str,
            "locked": bool(st.locked) if st else False,
            "live": bool(st.live) if st else False,
            "members": list(st.members) if st else [],
            "bound_id": st.id if st else None,
            "alone": track_pack(_logits(bundle, last_h), ids),
            "bound": track_pack(_logits(bundle, bound_h), ids),
            "ablate_random": track_pack(_logits(bundle, rand_h), ids),
            "bound_mid": bound_mid,
        }
        rows.append(row)
        payload = {
            "model": getattr(bundle, "name", model),
            "decay": decay,
            "gain": gain,
            "partial": True,
            "summary": mine(rows),
            "rows": rows,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"i={i} locked={row['locked']} alone_t={row['alone']['track']:.5f} "
            f"bound_t={row['bound']['track']:.5f} rand_t={row['ablate_random']['track']:.5f}",
            file=sys.stderr,
            flush=True,
        )

    # Shuffle: each prompt gets the next prompt's bound vector at the mouth.
    by_i = {int(r["i"]): r for r in rows}
    for i, (a, op, b, result) in enumerate(ITEMS):
        if i not in by_i:
            continue
        src = (i + 1) % len(ITEMS)
        last_h, _ = cached[i] if cached[i][0] is not None else (None, None)
        _, src_vec = cached[src] if cached[src][0] is not None else (None, None)
        if last_h is None:
            continue
        rid, _ = result_token_id(bundle, result)
        aid, _ = result_token_id(bundle, a)
        bid, _ = result_token_id(bundle, b)
        wrong_n = a + 1 if a + 1 != b else a + 2
        wid, _ = result_token_id(bundle, wrong_n)
        ids = {"a": aid, "b": bid, "wrong": wid, "cat": cat_id, "result": rid}
        if src_vec is None:
            shuf_h = last_h
        else:
            shuf_h = last_h + float(gain) * src_vec.to(dtype=last_h.dtype)
        by_i[i]["ablate_shuffle"] = track_pack(_logits(bundle, shuf_h), ids)

    rows = [by_i[i] for i in sorted(by_i)]
    summary = mine(rows)
    payload = {
        "model": getattr(bundle, "name", model),
        "decay": decay,
        "gain": gain,
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
    p.add_argument("-o", "--output", default="artifacts/bound_state_ablation_27b.json")
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
