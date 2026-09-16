"""Raise written answers by making the mouth use the live file.

Two conditions vs reused vanilla:

  file     live state is written into the prompt (file is the reasoner)
  gate     first live/rival token prefers the file (lead cup 3 → 1)

Not +gain on every step. Decay is not trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raise_reason_27b import ITEMS
from world_reason_27b import assess


def mine(rows: list[dict]) -> dict:
    n = len(rows)
    alone = sum(1 for r in rows if r["alone"]["wrote_result"])
    file_n = sum(1 for r in rows if (r.get("file") or {}).get("wrote_result"))
    gate_n = sum(1 for r in rows if (r.get("gate") or {}).get("wrote_result"))
    findings = [
        f"Correct: alone {alone}/{n}, file-in-prompt {file_n}/{n}, state-gate {gate_n}/{n}.",
    ]
    if file_n > alone or gate_n > alone:
        findings.append("Written answers rose versus vanilla.")
    else:
        findings.append("Neither condition raised written answers.")
    return {
        "n": n,
        "alone_correct": alone,
        "file_correct": file_n,
        "gate_correct": gate_n,
        "reasoning_raised": bool(file_n > alone or gate_n > alone),
        "findings": findings,
    }


def run(*, model: str, out: Path, load_in_4bit: bool | None, max_new_tokens: int, reuse_alone: Path | None) -> dict:
    from llmintent.forward import encode_prompt_ids
    from llmintent.predictbind import greedy_generate, lookup_token_id
    from llmintent.suite import load_suite_model, resolve_model_spec
    from llmintent.worldfile import apply_world

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(f"loading {model} 4bit={bool(four)} world-raise", file=sys.stderr, flush=True)
    bundle = load_suite_model(model=model, load_in_4bit=four)
    out.parent.mkdir(parents=True, exist_ok=True)
    reused: dict[str, str] = {}
    if reuse_alone and reuse_alone.exists():
        prev = json.loads(reuse_alone.read_text(encoding="utf-8"))
        reused = {r["id"]: r.get("alone_text") or "" for r in prev.get("rows") or []}

    rows: list[dict] = []
    for item in ITEMS:
        world = apply_world(item["text"])
        live = world.value_surfaces()
        rivals = world.rival_surfaces()
        live_ids = [i for s in live if (i := lookup_token_id(bundle.tokenizer, s)) is not None]
        rival_ids = [i for s in rivals if (i := lookup_token_id(bundle.tokenizer, s)) is not None]
        live_ids = list(dict.fromkeys(live_ids))
        rival_ids = [i for i in dict.fromkeys(rival_ids) if i not in live_ids]
        ids = encode_prompt_ids(bundle, item["text"], thinking=False)
        live_txt = world.format_live()
        file_prompt = f"{item['text']}\n\nLive entity state: {live_txt}."
        ids_file = encode_prompt_ids(bundle, file_prompt, thinking=False)
        print(f"{item['id']} live={live} rivals={rivals} file='{live_txt}'", file=sys.stderr, flush=True)

        if item["id"] in reused and reused[item["id"]]:
            alone_txt = reused[item["id"]]
        else:
            alone_txt = greedy_generate(bundle, ids, max_new_tokens=max_new_tokens, mode="alone")
        print(f"{item['id']} gen-file", file=sys.stderr, flush=True)
        file_txt = greedy_generate(bundle, ids_file, max_new_tokens=max_new_tokens, mode="alone")
        print(f"{item['id']} gen-gate", file=sys.stderr, flush=True)
        gate_txt = greedy_generate(
            bundle,
            ids,
            max_new_tokens=max_new_tokens,
            mode="alone",
            state_gate=True,
            live_ids=live_ids,
            rival_ids=rival_ids,
        )
        row = {
            "id": item["id"],
            "expect": item["expect"],
            "live": live,
            "rivals": rivals,
            "format_live": live_txt,
            "alone_text": alone_txt,
            "file_text": file_txt,
            "gate_text": gate_txt,
            "alone": assess(alone_txt, item),
            "file": assess(file_txt, item),
            "gate": assess(gate_txt, item),
        }
        rows.append(row)
        out.write_text(
            json.dumps({"partial": True, "summary": mine(rows), "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(
            f"{item['id']} alone={row['alone']['wrote_result']} "
            f"file={row['file']['wrote_result']} gate={row['gate']['wrote_result']}",
            file=sys.stderr,
            flush=True,
        )

    summary = mine(rows)
    payload = {
        "model": getattr(bundle, "name", model),
        "site": "world_file_visible_and_state_gate",
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
    p.add_argument("-o", "--output", default="artifacts/world_raise_27b.json")
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
