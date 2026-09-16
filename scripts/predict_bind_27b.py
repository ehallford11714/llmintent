"""Bind locked state into next-token prediction, then ablate.

Gap: residual and in-stream writes left the mouth on vanilla
lm_head(last). This scores the same hidden four ways:

  alone            transformer logits
  residual         last + g  (old path; not prediction)
  unembed          logits + gain * (W @ g)
  members          logits[member ids] += gain
  both             unembed + members
  shuffle_*        next prompt's g or members at the same mouth

Tracking = P(a)+P(b). One forward per prompt. Decay is not trained.
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


def _p(logits: torch.Tensor, tid: int | None) -> float:
    if tid is None:
        return 0.0
    return float(F.softmax(logits.float().reshape(-1), dim=-1)[int(tid)].item())


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
        "top_id": int(torch.argmax(logits.reshape(-1)).item()),
    }


def mine(rows: list[dict]) -> dict:
    def mean(cond: str, key: str) -> float:
        vals = [
            r[cond][key]
            for r in rows
            if cond in r and isinstance(r.get(cond), dict) and key in r[cond]
        ]
        return float(np.mean(vals)) if vals else 0.0

    n = len(rows) or 1
    locked = sum(1 for r in rows if r.get("locked"))
    alone = mean("alone", "track")
    residual = mean("residual", "track")
    unembed = mean("unembed", "track")
    members = mean("members", "track")
    both = mean("both", "track")
    sh_u = mean("shuffle_unembed", "track")
    sh_m = mean("shuffle_members", "track")
    floor = 1e-5
    rides_unembed = unembed > alone + floor and unembed > sh_u + floor
    rides_members = members > alone + floor and members > sh_m + floor
    rides_both = both > alone + floor and both > sh_m + floor
    findings = [
        f"BoundState locked on {locked}/{len(rows)}.",
        (
            f"Track P(a)+P(b): alone {alone:.5f}, residual {residual:.5f}, "
            f"unembed {unembed:.5f}, members {members:.5f}, both {both:.5f}, "
            f"shuffle_unembed {sh_u:.5f}, shuffle_members {sh_m:.5f}."
        ),
    ]
    if rides_members:
        findings.append("Member bind raised operand mass over alone and shuffled members.")
    else:
        findings.append("Member bind did not uniquely raise operand mass.")
    if rides_unembed:
        findings.append("Unembed bind raised operand mass over alone and shuffled g.")
    else:
        findings.append(
            "Projecting g through the unembed did not uniquely raise operand mass."
        )
    return {
        "n": len(rows),
        "locked_frac": round(locked / n, 3),
        "track_alone": round(alone, 6),
        "track_residual": round(residual, 6),
        "track_unembed": round(unembed, 6),
        "track_members": round(members, 6),
        "track_both": round(both, 6),
        "track_shuffle_unembed": round(sh_u, 6),
        "track_shuffle_members": round(sh_m, 6),
        "pref_alone": round(mean("alone", "pref"), 6),
        "pref_members": round(mean("members", "pref"), 6),
        "pref_unembed": round(mean("unembed", "pref"), 6),
        "rides_unembed": bool(rides_unembed),
        "rides_members": bool(rides_members),
        "rides_both": bool(rides_both),
        "findings": findings,
    }


def run(
    *,
    model: str,
    out: Path,
    load_in_4bit: bool | None,
    decay: float,
    gain_unembed: float,
    gain_members: float,
    gain_residual: float,
) -> dict:
    from llmintent.entity import detect_entity_spans
    from llmintent.forward import forward_hidden_states_from_ids, normalize_hidden
    from llmintent.persistbind import PersistenceBinder, drive_from_hidden
    from llmintent.predictbind import (
        apply_member_bind,
        apply_unembed_bind,
        bound_vocab_logits,
        member_token_ids,
        transformer_logits,
    )
    from llmintent.spike import SpikeConfig
    from llmintent.suite import load_suite_model, resolve_model_spec

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(
        f"loading {model} 4bit={bool(four)} items={len(ITEMS)} predict-bind",
        file=sys.stderr,
        flush=True,
    )
    bundle = load_suite_model(model=model, load_in_4bit=four)
    from llmintent.predictbind import lookup_token_id

    cat_id = lookup_token_id(bundle.tokenizer, "cat")
    if cat_id is None:
        cat_id = int(bundle.tokenizer.encode("cat", add_special_tokens=False)[0])

    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    extras: list[torch.Tensor | None] = [None] * len(ITEMS)
    mem_ids: list[list[int]] = [[] for _ in ITEMS]
    last_logits: list[torch.Tensor | None] = [None] * len(ITEMS)
    pack_ids: list[dict | None] = [None] * len(ITEMS)
    cfg = SpikeConfig(decay=decay)

    def result_tid(n: int) -> int | None:
        return lookup_token_id(bundle.tokenizer, str(n))

    for i, (a, op, b, result) in enumerate(ITEMS):
        text = f"{a} {op} {b}. The answer is"
        tok_ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
        last_str = bundle.tokenizer.decode(
            [int(tok_ids[0, -1].item())], skip_special_tokens=True
        )
        states = forward_hidden_states_from_ids(bundle, tok_ids)
        tokens = [
            bundle.tokenizer.decode([int(t)], skip_special_tokens=True).strip()
            for t in tok_ids[0].tolist()
        ]
        spans = detect_entity_spans(bundle.tokenizer, tok_ids)
        binder = PersistenceBinder(cfg)
        drive_from_hidden(binder, states, tokens=tokens, spans=spans)
        last_h = normalize_hidden(bundle, states[-1][0, -1, :].float())
        vec = binder.state_vector()
        st = binder.bound_state
        members = list(st.members) if st else []
        ids = {
            "a": result_tid(a),
            "b": result_tid(b),
            "wrong": result_tid(a + 1 if a + 1 != b else a + 2),
            "cat": cat_id,
            "result": result_tid(result),
        }
        mids = member_token_ids(bundle, members)
        logits0 = transformer_logits(bundle, last_h)
        extra = None if vec is None else bound_vocab_logits(bundle, vec)
        extras[i] = extra
        mem_ids[i] = mids
        last_logits[i] = logits0
        pack_ids[i] = ids

        residual_logits = logits0
        if vec is not None:
            steered = last_h.float().cpu().reshape(-1) + float(gain_residual) * vec.cpu().reshape(-1)
            residual_logits = transformer_logits(bundle, steered)

        unembed_logits = (
            apply_unembed_bind(logits0, extra, gain=gain_unembed) if extra is not None else logits0
        )
        member_logits = apply_member_bind(logits0, mids, gain=gain_members)
        both_logits = apply_member_bind(unembed_logits, mids, gain=gain_members)

        row = {
            "i": i,
            "text": text,
            "a": a,
            "b": b,
            "op": op,
            "result": result,
            "last_str": last_str,
            "locked": bool(st.locked) if st else False,
            "member_names": members,
            "member_ids": mids,
            "alone": track_pack(logits0, ids),
            "residual": track_pack(residual_logits, ids),
            "unembed": track_pack(unembed_logits, ids),
            "members": track_pack(member_logits, ids),
            "both": track_pack(both_logits, ids),
        }
        rows.append(row)
        out.write_text(
            json.dumps(
                {
                    "model": getattr(bundle, "name", model),
                    "decay": decay,
                    "gain_unembed": gain_unembed,
                    "gain_members": gain_members,
                    "gain_residual": gain_residual,
                    "partial": True,
                    "summary": mine(rows),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"i={i} locked={row['locked']} names={members} "
            f"alone={row['alone']['track']:.5f} "
            f"unembed={row['unembed']['track']:.5f} "
            f"member={row['members']['track']:.5f}",
            file=sys.stderr,
            flush=True,
        )

    for i, row in enumerate(rows):
        src = (i + 1) % len(ITEMS)
        base = last_logits[i]
        ids = pack_ids[i]
        if base is None or ids is None:
            continue
        src_extra = extras[src]
        src_mids = mem_ids[src]
        sh_u = (
            apply_unembed_bind(base, src_extra, gain=gain_unembed)
            if src_extra is not None
            else base
        )
        sh_m = apply_member_bind(base, src_mids, gain=gain_members)
        row["shuffle_unembed"] = track_pack(sh_u, ids)
        row["shuffle_members"] = track_pack(sh_m, ids)
        print(
            f"i={i} shuffle_u={row['shuffle_unembed']['track']:.5f} "
            f"shuffle_m={row['shuffle_members']['track']:.5f}",
            file=sys.stderr,
            flush=True,
        )

    summary = mine(rows)
    payload = {
        "model": getattr(bundle, "name", model),
        "decay": decay,
        "gain_unembed": gain_unembed,
        "gain_members": gain_members,
        "gain_residual": gain_residual,
        "site": "lm_head_prediction",
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
    p.add_argument("--gain-unembed", type=float, default=1.5)
    p.add_argument("--gain-members", type=float, default=8.0)
    p.add_argument("--gain-residual", type=float, default=1.5)
    p.add_argument("-o", "--output", default="artifacts/predict_bind_27b.json")
    args = p.parse_args(argv)
    run(
        model=args.model,
        out=Path(args.output),
        load_in_4bit=True if args.load_in_4bit else None,
        decay=args.decay,
        gain_unembed=args.gain_unembed,
        gain_members=args.gain_members,
        gain_residual=args.gain_residual,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
