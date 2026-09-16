"""Inform the mouth with the *current* world file, not the name list.

Reasoning here is entity persistence: the live situation after the last
update. Vanilla dropped the last cup swap. Name-lock stored {1,2,3}.
This prior is the ball's cup now.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raise_reason_27b import ITEMS, _has_word, verdict


def mine(rows: list[dict]) -> dict:
    n = len(rows)
    alone_ok = sum(1 for r in rows if (r.get("alone") or {}).get("wrote_result"))
    raise_ok = sum(1 for r in rows if (r.get("raise") or {}).get("wrote_result"))
    counts: dict[str, int] = {}
    for r in rows:
        v = r.get("verdict") or "same"
        counts[v] = counts.get(v, 0) + 1
    findings = [
        f"Lead/state-correct: alone {alone_ok}/{n}, world-file {raise_ok}/{n}.",
        f"Verdicts: {counts}.",
    ]
    if raise_ok > alone_ok:
        findings.append("Current-state prior raised persistence marks.")
    elif raise_ok < alone_ok:
        findings.append("Current-state prior lost marks versus vanilla.")
    else:
        findings.append("No net gain versus vanilla.")
    return {
        "n": n,
        "alone_correct": alone_ok,
        "raise_correct": raise_ok,
        "verdicts": counts,
        "reasoning_raised": bool(raise_ok > alone_ok),
        "findings": findings,
    }


def assess(text: str, item: dict) -> dict:
    t = (text or "").strip()
    expect = list(item.get("expect") or [])
    wrote = all(_has_word(t, e) for e in expect) if expect else False
    lead = None
    if item["id"] == "shell_3":
        m = re.search(r"cup\s+(\d+)", t, re.I)
        lead = m.group(1) if m else None
        wrote = lead == "1"
    return {
        "empty": not t,
        "n_chars": len(t),
        "wrote_result": wrote,
        "lead": lead,
        "preview": t[:280],
    }


def run(
    *,
    model: str,
    out: Path,
    load_in_4bit: bool | None,
    max_new_tokens: int,
    reuse_alone: Path | None,
) -> dict:
    from llmintent.forward import encode_prompt_ids
    from llmintent.predictbind import greedy_generate, lookup_token_id
    from llmintent.suite import load_suite_model, resolve_model_spec
    from llmintent.worldfile import apply_world

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(
        f"loading {model} 4bit={bool(four)} items={len(ITEMS)} world-file",
        file=sys.stderr,
        flush=True,
    )
    bundle = load_suite_model(model=model, load_in_4bit=four)
    out.parent.mkdir(parents=True, exist_ok=True)
    reused: dict[str, str] = {}
    if reuse_alone is not None and reuse_alone.exists():
        prev = json.loads(reuse_alone.read_text(encoding="utf-8"))
        reused = {r["id"]: r["alone_text"] for r in prev.get("rows") or [] if r.get("alone_text")}

    rows: list[dict] = []
    for item in ITEMS:
        world = apply_world(item["text"])
        surfaces = world.surfaces()
        mids = []
        for s in surfaces:
            tid = lookup_token_id(bundle.tokenizer, s)
            if tid is not None and tid not in mids:
                mids.append(tid)
        ids = encode_prompt_ids(bundle, item["text"], thinking=False)
        print(
            f"{item['id']} state={surfaces} loc={world.loc} count={world.count} "
            f"acc={world.acc} gen-alone{' (reuse)' if item['id'] in reused else ''}",
            file=sys.stderr,
            flush=True,
        )
        if item["id"] in reused:
            alone_txt = reused[item["id"]]
        else:
            alone_txt = greedy_generate(bundle, ids, max_new_tokens=max_new_tokens, mode="alone")
        print(f"{item['id']} gen-world", file=sys.stderr, flush=True)
        world_txt = greedy_generate(
            bundle,
            ids,
            max_new_tokens=max_new_tokens,
            member_ids=mids,
            mode="bayes",
            query_only=False,
        )
        alone_a = assess(alone_txt, item)
        world_a = assess(world_txt, item)
        row = {
            "id": item["id"],
            "text": item["text"],
            "expect": item["expect"],
            "surfaces": surfaces,
            "loc": world.loc,
            "count": world.count,
            "acc": world.acc,
            "alone_text": alone_txt,
            "world_text": world_txt,
            "alone": alone_a,
            "raise": world_a,
            "verdict": verdict(alone_a, world_a),
        }
        rows.append(row)
        out.write_text(
            json.dumps(
                {
                    "model": getattr(bundle, "name", model),
                    "partial": True,
                    "summary": mine(rows),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"{item['id']} verdict={row['verdict']} "
            f"alone_ok={alone_a['wrote_result']} world_ok={world_a['wrote_result']}",
            file=sys.stderr,
            flush=True,
        )

    summary = mine(rows)
    payload = {
        "model": getattr(bundle, "name", model),
        "site": "world_file_current_state",
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
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--reuse-alone", default="artifacts/raise_reason_27b.json")
    p.add_argument("-o", "--output", default="artifacts/world_reason_27b.json")
    args = p.parse_args(argv)
    run(
        model=args.model,
        out=Path(args.output),
        load_in_4bit=True if args.load_in_4bit else None,
        max_new_tokens=args.max_new,
        reuse_alone=Path(args.reuse_alone) if args.reuse_alone else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
