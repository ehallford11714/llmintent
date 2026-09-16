"""Bind locked state *inside* the transformer residual stream, then ablate.

Pass 1: vanilla forward, lock BoundState.
Pass 2: every block writes gain/n_layers * g at member sites and the tail.
Ablate: shuffle another prompt's g through the same in-stream write.

Tracking = P(a)+P(b) at the mouth. Not a number-box claim.
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
    return int(ids[0]) if len(ids) == 1 else None


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

        return F.linear(h.cpu(), get_unembedding_matrix(bundle.model).float().cpu()).float()


def _p(logits: torch.Tensor, tid: int) -> float:
    return float(F.softmax(logits.float(), dim=-1)[tid].item())


def track_pack(logits: torch.Tensor, ids: dict[str, int]) -> dict:
    p_a, p_b = _p(logits, ids["a"]), _p(logits, ids["b"])
    p_w, p_c = _p(logits, ids["wrong"]), _p(logits, ids["cat"])
    track = p_a + p_b
    return {
        "p_a": round(p_a, 6),
        "p_b": round(p_b, 6),
        "p_wrong": round(p_w, 6),
        "p_cat": round(p_c, 6),
        "p_result": round(_p(logits, ids["result"]), 6),
        "track": round(track, 6),
        "pref": round(track - p_w - p_c, 6),
        "top_id": int(torch.argmax(logits).item()),
    }


def mine(rows: list[dict]) -> dict:
    def mean(cond: str, key: str) -> float:
        vals = [r[cond][key] for r in rows if cond in r and isinstance(r[cond], dict) and key in r[cond]]
        return float(np.mean(vals)) if vals else 0.0

    n = len(rows) or 1
    locked = sum(1 for r in rows if r.get("locked"))
    alone, within, shuf = mean("alone", "track"), mean("within", "track"), mean("shuffle", "track")
    rides = within > alone + 1e-5 and within > shuf + 1e-5
    findings = [
        f"BoundState locked on {locked}/{len(rows)}.",
        f"In-stream track P(a)+P(b): alone {alone:.5f}, within {within:.5f}, shuffle {shuf:.5f}.",
    ]
    if rides:
        findings.append("In-stream bind raised operand mass over alone and shuffle.")
    else:
        findings.append(
            "Writing the lock into the residual stream did not uniquely raise operand mass."
        )
    return {
        "n": len(rows),
        "locked_frac": round(locked / n, 3),
        "track_alone": round(alone, 6),
        "track_within": round(within, 6),
        "track_shuffle": round(shuf, 6),
        "pref_alone": round(mean("alone", "pref"), 6),
        "pref_within": round(mean("within", "pref"), 6),
        "pref_shuffle": round(mean("shuffle", "pref"), 6),
        "rides_bind": bool(rides),
        "findings": findings,
    }


def run(*, model: str, out: Path, load_in_4bit: bool | None, decay: float, gain: float) -> dict:
    from llmintent.entity import detect_entity_spans
    from llmintent.forward import forward_hidden_states_from_ids, normalize_hidden
    from llmintent.instream import bind_positions, forward_within
    from llmintent.persistbind import PersistenceBinder, drive_from_hidden
    from llmintent.spike import SpikeConfig
    from llmintent.suite import load_suite_model, resolve_model_spec

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(f"loading {model} 4bit={bool(four)} items={len(ITEMS)} within-stream", file=sys.stderr, flush=True)
    bundle = load_suite_model(model=model, load_in_4bit=four)
    cat_id = _tok_id(bundle, "cat") or _tok_id(bundle, " cat")
    if cat_id is None:
        cat_id = int(bundle.tokenizer.encode("cat", add_special_tokens=False)[0])

    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    vecs: list[torch.Tensor | None] = [None] * len(ITEMS)
    pos_list: list[list[int]] = [[] for _ in ITEMS]
    id_list: list[torch.Tensor | None] = [None] * len(ITEMS)
    pack_ids: list[dict | None] = [None] * len(ITEMS)
    cfg = SpikeConfig(decay=decay)

    for i, (a, op, b, result) in enumerate(ITEMS):
        text = f"{a} {op} {b}. The answer is"
        tok_ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
        last_str = _piece(bundle.tokenizer, int(tok_ids[0, -1].item()))
        states = forward_hidden_states_from_ids(bundle, tok_ids)
        tokens = [_piece(bundle.tokenizer, int(t)).strip() for t in tok_ids[0].tolist()]
        spans = detect_entity_spans(bundle.tokenizer, tok_ids)
        binder = PersistenceBinder(cfg)
        drive_from_hidden(binder, states, tokens=tokens, spans=spans)
        last_h = normalize_hidden(bundle, states[-1][0, -1, :].float())
        vec = binder.state_vector()
        pos = bind_positions(int(tok_ids.shape[1]), spans)
        vecs[i] = None if vec is None else vec.detach().cpu().clone()
        pos_list[i] = pos
        id_list[i] = tok_ids
        ids = {
            "a": result_token_id(bundle, a)[0],
            "b": result_token_id(bundle, b)[0],
            "wrong": result_token_id(bundle, a + 1 if a + 1 != b else a + 2)[0],
            "cat": cat_id,
            "result": result_token_id(bundle, result)[0],
        }
        pack_ids[i] = ids
        within = None
        if vec is not None:
            states_w = forward_within(bundle, tok_ids, vec.to(bundle.device), pos, gain=gain)
            h_w = normalize_hidden(bundle, states_w[-1][0, -1, :].float())
            within = track_pack(_logits(bundle, h_w), ids)
        st = binder.bound_state
        row = {
            "i": i,
            "text": text,
            "a": a,
            "b": b,
            "last_str": last_str,
            "locked": bool(st.locked) if st else False,
            "members": list(st.members) if st else [],
            "positions": pos,
            "alone": track_pack(_logits(bundle, last_h), ids),
            "within": within,
        }
        rows.append(row)
        out.write_text(
            json.dumps(
                {
                    "model": getattr(bundle, "name", model),
                    "gain": gain,
                    "decay": decay,
                    "partial": True,
                    "summary": mine(rows),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        wt = None if within is None else within["track"]
        print(
            f"i={i} locked={row['locked']} pos={pos} "
            f"alone={row['alone']['track']:.5f} within={wt}",
            file=sys.stderr,
            flush=True,
        )

    for i, row in enumerate(rows):
        src = (i + 1) % len(ITEMS)
        src_vec = vecs[src]
        tok_ids = id_list[i]
        ids = pack_ids[i]
        if src_vec is None or tok_ids is None or ids is None:
            continue
        states_s = forward_within(
            bundle, tok_ids, src_vec.to(bundle.device), pos_list[i], gain=gain
        )
        from llmintent.forward import normalize_hidden as _norm

        h_s = _norm(bundle, states_s[-1][0, -1, :].float())
        row["shuffle"] = track_pack(_logits(bundle, h_s), ids)
        print(f"i={i} shuffle={row['shuffle']['track']:.5f}", file=sys.stderr, flush=True)

    summary = mine(rows)
    payload = {
        "model": getattr(bundle, "name", model),
        "gain": gain,
        "decay": decay,
        "site": "all_layers_member_and_tail",
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
    p.add_argument("-o", "--output", default="artifacts/bind_within_27b.json")
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
