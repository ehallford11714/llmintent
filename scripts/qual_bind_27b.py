"""Qualitative: does prediction-bind improve reasoning, or just operand mass?

Chat-template greedy decode. Thinking off. Two conditions from one lock:

  alone     vanilla generate
  bound     lock at the mouth (default: Bayesian inform, not +gain)

Rubric is flags plus a side-by-side verdict. Not a number-box claim.
Decay is not trained.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ITEMS = [
    {
        "id": "arith_8_2",
        "kind": "arith",
        "text": "What is 8 minus 2? Show the numbers you are using, then give the result.",
        "expect": ["6"],
        "operands": ["8", "2"],
    },
    {
        "id": "arith_7_5",
        "kind": "arith",
        "text": "Compute 7 plus 5. Explain briefly, then give the sum.",
        "expect": ["12"],
        "operands": ["7", "5"],
    },
    {
        "id": "alice_apples",
        "kind": "update",
        "text": "Alice has 8 apples. She gives 2 apples to Bob. How many apples does Alice have now?",
        "expect": ["6"],
        "operands": ["8", "2"],
    },
    {
        "id": "box_count",
        "kind": "multi",
        "text": "A box has 4 stones. You add 3 more, then take 1 out. How many stones are in the box?",
        "expect": ["6"],
        "operands": ["4", "3"],
    },
    {
        "id": "cat_move",
        "kind": "track",
        "text": "The cat is on the river. Then the cat moves to the tree. Where is the cat now?",
        "expect": ["tree"],
        "operands": ["cat", "river", "tree"],
    },
    {
        "id": "swap_bags",
        "kind": "swap",
        "text": (
            "Maya puts a cat in the red bag and a dog in the blue bag. "
            "She swaps them. What is in the red bag now?"
        ),
        "expect": ["dog"],
        "operands": ["cat", "dog"],
    },
    {
        "id": "same_op",
        "kind": "arith",
        "text": "What is 6 plus 6? Give the sum and a one-sentence reason.",
        "expect": ["12"],
        "operands": ["6"],
    },
    {
        "id": "ignore_7",
        "kind": "arith",
        "text": "Solve 9 times 2. Do not use the number 7. Give the product.",
        "expect": ["18"],
        "operands": ["9", "2"],
    },
]

_REASON = re.compile(
    r"\b(because|therefore|so |first|then|minus|plus|times|equals?|"
    r"left|now|sum|product|difference|moves?|swaps?|gives?)\b",
    re.I,
)


def _has_word(text: str, piece: str) -> bool:
    if not piece:
        return False
    if piece.isdigit():
        return re.search(rf"(?<!\d){re.escape(piece)}(?!\d)", text) is not None
    return re.search(rf"\b{re.escape(piece)}\b", text, re.I) is not None


def assess(text: str, item: dict) -> dict:
    raw = text or ""
    t = raw.strip()
    expect = list(item.get("expect") or [])
    ops = list(item.get("operands") or [])
    wrote_result = any(_has_word(t, e) for e in expect)
    wrote_ops = all(_has_word(t, o) for o in ops) if ops else False
    first = t.split()[0] if t.split() else ""
    first_clean = re.sub(r"[^\w]", "", first)
    hijack = bool(t) and first_clean in set(ops) and not _REASON.search(t[:80])
    return {
        "empty": not t,
        "n_chars": len(t),
        "wrote_result": wrote_result,
        "wrote_operands": wrote_ops,
        "has_reason": bool(_REASON.search(t)),
        "opens_with_operand": first_clean in set(ops),
        "hijack": hijack,
        "preview": t[:240],
    }


def verdict(alone: dict, bound: dict) -> str:
    if bound.get("hijack") and not alone.get("hijack"):
        if bound.get("wrote_result") and not alone.get("wrote_result"):
            return "hijack_but_correct"
        return "hijack"
    if bound.get("wrote_result") and not alone.get("wrote_result"):
        if bound.get("has_reason") or bound.get("n_chars", 0) >= 40:
            return "improved"
        return "correct_shorter"
    if alone.get("wrote_result") and not bound.get("wrote_result"):
        return "worse"
    if alone.get("has_reason") and not bound.get("has_reason"):
        return "worse_reason"
    if bound.get("has_reason") and not alone.get("has_reason") and bound.get("wrote_result"):
        return "improved"
    return "same"


def mine(rows: list[dict]) -> dict:
    n = len(rows) or 1
    counts: dict[str, int] = {}
    for r in rows:
        v = r.get("verdict") or "same"
        counts[v] = counts.get(v, 0) + 1
    alone_ok = sum(1 for r in rows if (r.get("alone") or {}).get("wrote_result"))
    bind_ok = sum(1 for r in rows if (r.get("members") or {}).get("wrote_result"))
    hijack = sum(1 for r in rows if (r.get("members") or {}).get("hijack"))
    improved = counts.get("improved", 0) + counts.get("correct_shorter", 0)
    worse = counts.get("worse", 0) + counts.get("worse_reason", 0) + counts.get("hijack", 0)
    findings = [
        f"Correct answer present: alone {alone_ok}/{len(rows)}, members {bind_ok}/{len(rows)}.",
        f"Verdicts: {counts}.",
        f"Member-bind hijack (opens on a locked name, no reason cue): {hijack}/{len(rows)}.",
    ]
    if bind_ok > alone_ok and improved > worse:
        findings.append("Member bind improved reasoning on this set.")
    elif bind_ok < alone_ok or worse > improved:
        findings.append("Member bind did not improve reasoning; it often hijacked the first token.")
    else:
        findings.append("No clear reasoning gain. Operand mass is not the same as better work.")
    return {
        "n": len(rows),
        "alone_correct": alone_ok,
        "members_correct": bind_ok,
        "hijack": hijack,
        "verdicts": counts,
        "improved": improved,
        "worse": worse,
        "same": counts.get("same", 0),
        "reasoning_improved": bool(bind_ok > alone_ok and improved > worse),
        "findings": findings,
    }


def run(
    *,
    model: str,
    out: Path,
    load_in_4bit: bool | None,
    decay: float,
    gain_members: float,
    max_new_tokens: int,
    bind_mode: str = "bayes",
    reuse_alone: Path | None = None,
    member_mass: float = 0.02,
    max_strength: float = 1.0,
) -> dict:
    from llmintent.entity import detect_entity_spans
    from llmintent.forward import encode_prompt_ids, forward_hidden_states_from_ids
    from llmintent.persistbind import PersistenceBinder, drive_from_hidden
    from llmintent.predictbind import greedy_generate, member_token_ids
    from llmintent.spike import SpikeConfig
    from llmintent.suite import load_suite_model, resolve_model_spec

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(
        f"loading {model} 4bit={bool(four)} items={len(ITEMS)} qual-bind mode={bind_mode}",
        file=sys.stderr,
        flush=True,
    )
    bundle = load_suite_model(model=model, load_in_4bit=four)
    cfg = SpikeConfig(decay=decay)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    reused: dict[str, str] = {}
    if reuse_alone is not None and reuse_alone.exists():
        prev = json.loads(reuse_alone.read_text(encoding="utf-8"))
        reused = {r["id"]: r["alone_text"] for r in prev.get("rows") or [] if r.get("alone_text")}

    for item in ITEMS:
        ids = encode_prompt_ids(bundle, item["text"], thinking=False)
        states = forward_hidden_states_from_ids(bundle, ids)
        tokens = [
            bundle.tokenizer.decode([int(t)], skip_special_tokens=True).strip()
            for t in ids[0].tolist()
        ]
        spans = detect_entity_spans(bundle.tokenizer, ids)
        binder = PersistenceBinder(cfg)
        drive_from_hidden(binder, states, tokens=tokens, spans=spans)
        st = binder.bound_state
        members = list(st.members) if st else []
        mids = member_token_ids(bundle, members)
        print(
            f"{item['id']} lock={bool(st.locked) if st else False} names={members} "
            f"gen-alone{' (reuse)' if item['id'] in reused else ''}",
            file=sys.stderr,
            flush=True,
        )
        if item["id"] in reused:
            alone_txt = reused[item["id"]]
        else:
            alone_txt = greedy_generate(bundle, ids, max_new_tokens=max_new_tokens, mode="alone")
        print(f"{item['id']} gen-{bind_mode}", file=sys.stderr, flush=True)
        bind_txt = greedy_generate(
            bundle,
            ids,
            max_new_tokens=max_new_tokens,
            members=members,
            member_ids=mids,
            mode=bind_mode,
            gain_members=gain_members,
            member_mass=member_mass,
            max_strength=max_strength,
        )
        alone_a = assess(alone_txt, item)
        bind_a = assess(bind_txt, item)
        row = {
            "id": item["id"],
            "kind": item["kind"],
            "text": item["text"],
            "expect": item["expect"],
            "locked": bool(st.locked) if st else False,
            "member_names": members,
            "member_ids": mids,
            "alone_text": alone_txt,
            "members_text": bind_txt,
            "alone": alone_a,
            "members": bind_a,
            "verdict": verdict(alone_a, bind_a),
        }
        rows.append(row)
        payload = {
            "model": getattr(bundle, "name", model),
            "decay": decay,
            "gain_members": gain_members,
            "bind_mode": bind_mode,
            "member_mass": member_mass,
            "max_strength": max_strength,
            "max_new_tokens": max_new_tokens,
            "partial": True,
            "summary": mine(rows),
            "rows": rows,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"{item['id']} verdict={row['verdict']} "
            f"alone_ok={alone_a['wrote_result']} bind_ok={bind_a['wrote_result']}",
            file=sys.stderr,
            flush=True,
        )

    summary = mine(rows)
    payload = {
        "model": getattr(bundle, "name", model),
        "decay": decay,
        "gain_members": gain_members,
        "bind_mode": bind_mode,
        "member_mass": member_mass,
        "max_strength": max_strength,
        "max_new_tokens": max_new_tokens,
        "site": "chat_greedy_prediction_bind",
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
    p.add_argument("--gain-members", type=float, default=8.0)
    p.add_argument("--mode", dest="bind_mode", default="bayes")
    p.add_argument("--member-mass", type=float, default=0.02)
    p.add_argument("--max-strength", type=float, default=1.0)
    p.add_argument("--reuse-alone", default="")
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("-o", "--output", default="artifacts/qual_bind_27b.json")
    args = p.parse_args(argv)
    run(
        model=args.model,
        out=Path(args.output),
        load_in_4bit=True if args.load_in_4bit else None,
        decay=args.decay,
        gain_members=args.gain_members,
        max_new_tokens=args.max_new,
        bind_mode=args.bind_mode,
        reuse_alone=Path(args.reuse_alone) if args.reuse_alone else None,
        member_mass=args.member_mass,
        max_strength=args.max_strength,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
