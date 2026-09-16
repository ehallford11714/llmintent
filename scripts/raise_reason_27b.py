"""Raise reasoning: memory slot + query-only Bayes on hard tracking items.

Vanilla chat already solves one-step 8-2. This set is sequential state
change. Raise = lock grows with mentions, g is a token the stack attends
to, Bayes informs only the first generated token.

Not +gain. Not residual paint. Decay is not trained.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ITEMS = [
    {
        "id": "shell_3",
        "kind": "track",
        "text": (
            "A ball starts under cup 1. Swap cup 1 with cup 2. "
            "Swap cup 2 with cup 3. Swap cup 1 with cup 3. "
            "Which cup is the ball under? Give the cup number and one sentence."
        ),
        "expect": ["1"],
    },
    {
        "id": "two_move",
        "kind": "track",
        "text": (
            "The cat is on the river. The dog is in the house. "
            "The cat moves to the tree. The dog moves to the river. "
            "The cat moves to the house. Where is the cat now? Where is the dog now?"
        ),
        "expect": ["house", "river"],
    },
    {
        "id": "give_chain",
        "kind": "update",
        "text": (
            "Ada has 9 books. Bea has 4 books. Ada gives 3 to Bea. "
            "Bea gives 2 to Ada. Ada gives 1 to Bea. "
            "How many books does Ada have? How many does Bea have?"
        ),
        "expect": ["7", "6"],
    },
    {
        "id": "false_move",
        "kind": "track",
        "text": (
            "The egg is in the apple box. Someone says they moved it to the tree box, "
            "but that is false. It was actually moved to the car. Where is the egg?"
        ),
        "expect": ["car"],
    },
    {
        "id": "count_chain",
        "kind": "multi",
        "text": (
            "A box has 3 stones. Add 2. Take 1 out. Add 4. Take 3 out. "
            "How many stones are in the box?"
        ),
        "expect": ["5"],
    },
    {
        "id": "swap_then_box",
        "kind": "swap",
        "text": (
            "A cat is in the red bag and a dog is in the blue bag. "
            "They are swapped. Then they are swapped again. "
            "Then the cat is moved to the box. What is in the blue bag? What is in the box?"
        ),
        "expect": ["dog", "cat"],
    },
    {
        "id": "distract_num",
        "kind": "multi",
        "text": "Ignore 15. Start with 8. Subtract 2. Add 10. Subtract 3. What is the result?",
        "expect": ["13"],
    },
    {
        "id": "who_last",
        "kind": "track",
        "text": (
            "The book is held by the girl. She gives it to the boy. "
            "He gives it to the bird. The bird gives it to the fox. Who holds the book?"
        ),
        "expect": ["fox"],
    },
]

_REASON = re.compile(
    r"\b(because|therefore|so |first|then|minus|plus|times|equals?|"
    r"left|now|sum|under|holds?|moves?|swaps?|gives?)\b",
    re.I,
)


def _has_word(text: str, piece: str) -> bool:
    if not piece:
        return False
    if piece.isdigit():
        return re.search(rf"(?<!\d){re.escape(piece)}(?!\d)", text) is not None
    return re.search(rf"\b{re.escape(piece)}\b", text, re.I) is not None


def assess(text: str, item: dict) -> dict:
    t = (text or "").strip()
    expect = list(item.get("expect") or [])
    wrote = all(_has_word(t, e) for e in expect) if expect else False
    return {
        "empty": not t,
        "n_chars": len(t),
        "wrote_result": wrote,
        "has_reason": bool(_REASON.search(t)),
        "preview": t[:280],
    }


def verdict(alone: dict, raised: dict) -> str:
    if raised.get("wrote_result") and not alone.get("wrote_result"):
        return "improved"
    if alone.get("wrote_result") and not raised.get("wrote_result"):
        return "worse"
    return "same"


def mine(rows: list[dict]) -> dict:
    n = len(rows)
    alone_ok = sum(1 for r in rows if (r.get("alone") or {}).get("wrote_result"))
    raise_ok = sum(1 for r in rows if (r.get("raise") or {}).get("wrote_result"))
    counts: dict[str, int] = {}
    for r in rows:
        v = r.get("verdict") or "same"
        counts[v] = counts.get(v, 0) + 1
    findings = [
        f"Correct (all expected marks): alone {alone_ok}/{n}, raise {raise_ok}/{n}.",
        f"Verdicts: {counts}.",
    ]
    if raise_ok > alone_ok:
        findings.append("Memory slot + query Bayes raised correct answers.")
    elif raise_ok < alone_ok:
        findings.append("Raise condition lost answers versus vanilla.")
    else:
        findings.append("No net gain in correct marks versus vanilla.")
    return {
        "n": n,
        "alone_correct": alone_ok,
        "raise_correct": raise_ok,
        "verdicts": counts,
        "reasoning_raised": bool(raise_ok > alone_ok),
        "findings": findings,
    }


def run(
    *,
    model: str,
    out: Path,
    load_in_4bit: bool | None,
    decay: float,
    max_new_tokens: int,
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
        f"loading {model} 4bit={bool(four)} items={len(ITEMS)} raise-reason",
        file=sys.stderr,
        flush=True,
    )
    bundle = load_suite_model(model=model, load_in_4bit=four)
    cfg = SpikeConfig(decay=decay)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

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
        vec = binder.state_vector()
        mids = member_token_ids(bundle, members)
        print(
            f"{item['id']} lock={bool(st.locked) if st else False} "
            f"names={members} mentions={binder.mentions} gen-alone",
            file=sys.stderr,
            flush=True,
        )
        alone_txt = greedy_generate(bundle, ids, max_new_tokens=max_new_tokens, mode="alone")
        print(f"{item['id']} gen-raise", file=sys.stderr, flush=True)
        raise_txt = greedy_generate(
            bundle,
            ids,
            max_new_tokens=max_new_tokens,
            members=members,
            member_ids=mids,
            mode="bayes",
            query_only=True,
            memory_vector=vec,
            memory_gain=1.0,
        )
        alone_a = assess(alone_txt, item)
        raise_a = assess(raise_txt, item)
        row = {
            "id": item["id"],
            "kind": item["kind"],
            "text": item["text"],
            "expect": item["expect"],
            "locked": bool(st.locked) if st else False,
            "member_names": members,
            "mentions": binder.mentions,
            "alone_text": alone_txt,
            "raise_text": raise_txt,
            "alone": alone_a,
            "raise": raise_a,
            "verdict": verdict(alone_a, raise_a),
        }
        rows.append(row)
        payload = {
            "model": getattr(bundle, "name", model),
            "partial": True,
            "summary": mine(rows),
            "rows": rows,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"{item['id']} verdict={row['verdict']} "
            f"alone_ok={alone_a['wrote_result']} raise_ok={raise_a['wrote_result']}",
            file=sys.stderr,
            flush=True,
        )

    summary = mine(rows)
    payload = {
        "model": getattr(bundle, "name", model),
        "decay": decay,
        "max_new_tokens": max_new_tokens,
        "site": "memory_slot_query_bayes",
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
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("-o", "--output", default="artifacts/raise_reason_27b.json")
    args = p.parse_args(argv)
    run(
        model=args.model,
        out=Path(args.output),
        load_in_4bit=True if args.load_in_4bit else None,
        decay=args.decay,
        max_new_tokens=args.max_new,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
