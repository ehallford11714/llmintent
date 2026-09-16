"""Higher update-count persistence, then ablation.

Reasoning here is the current situation file. Vanilla reconstructs at the
query and can drop the last updates as the chain grows. The file should
not. These conditions ask whether the *mouth* still uses that file.

Experiment:  alone | file (live state in prompt) | gate (first live/rival token)

Ablation:    file_stale (drop last update) | file_wrong (rival as live)
             file_empty (label only)       | gate_wrong (live/rival swapped)

Ask for the number on the first line so a walkthrough-start is not a hit.
Decay is not trained. Not +gain. One 27B load.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ASK = " Answer with the number on the first line, then one sentence."
ASK2 = " Answer with both numbers on the first line, then one sentence."


def _shell(n: int, swaps: list[tuple[str, str]], expect: str) -> dict:
    bits = ["A ball starts under cup 1."]
    bits += [f"Swap cup {a} with cup {b}." for a, b in swaps]
    bits.append("Which cup is the ball under?")
    text = " ".join(bits)
    stale = " ".join(bits[:-2] + bits[-1:])  # drop last swap
    return {
        "id": f"shell_n{n}",
        "family": "shell",
        "n": n,
        "text": text + ASK,
        "stale_text": stale + ASK,
        "expect": [expect],
        "wrong": next(x for x in ("1", "2", "3") if x != expect),
    }


def _count(n: int, ops: list[str], expect: str) -> dict:
    body = "A box has 3 stones. " + " ".join(ops)
    text = body + " How many stones are in the box?"
    stale_ops = ops[:-1]
    stale = "A box has 3 stones. " + " ".join(stale_ops) + " How many stones are in the box?"
    return {
        "id": f"count_n{n}",
        "family": "count",
        "n": n,
        "text": text + ASK,
        "stale_text": stale + ASK,
        "expect": [expect],
        "wrong": "3",
    }


def _give(n: int, gives: list[str], expect: list[str]) -> dict:
    head = "Ada has 9 books. Bea has 4 books. "
    text = head + " ".join(gives) + " How many books does Ada have? How many does Bea have?"
    stale = head + " ".join(gives[:-1]) + " How many books does Ada have? How many does Bea have?"
    return {
        "id": f"give_n{n}",
        "family": "give",
        "n": n,
        "text": text + ASK2,
        "stale_text": stale + ASK2,
        "expect": expect,
        "wrong": "9",
    }


ITEMS = [
    _shell(3, [("1", "2"), ("2", "3"), ("1", "3")], "1"),
    _shell(
        7,
        [("1", "2"), ("2", "3"), ("1", "3"), ("1", "2"), ("2", "3"), ("1", "3"), ("2", "1")],
        "2",
    ),
    _shell(
        11,
        [
            ("1", "2"),
            ("2", "3"),
            ("1", "3"),
            ("1", "2"),
            ("2", "3"),
            ("1", "3"),
            ("2", "1"),
            ("2", "3"),
            ("1", "3"),
            ("1", "2"),
            ("2", "1"),
        ],
        "1",
    ),
    _count(5, ["Add 2.", "Take 1 out.", "Add 4.", "Take 3 out."], "5"),
    _count(
        9,
        [
            "Add 2.",
            "Take 1 out.",
            "Add 4.",
            "Take 3 out.",
            "Add 6.",
            "Take 2 out.",
            "Add 1.",
            "Take 4 out.",
        ],
        "6",
    ),
    _count(
        13,
        [
            "Add 2.",
            "Take 1 out.",
            "Add 4.",
            "Take 3 out.",
            "Add 6.",
            "Take 2 out.",
            "Add 1.",
            "Take 4 out.",
            "Add 5.",
            "Take 3 out.",
            "Add 2.",
            "Take 1 out.",
        ],
        "9",
    ),
    _give(3, ["Ada gives 3 to Bea.", "Bea gives 2 to Ada.", "Ada gives 1 to Bea."], ["7", "6"]),
    _give(
        9,
        [
            "Ada gives 3 to Bea.",
            "Bea gives 2 to Ada.",
            "Ada gives 1 to Bea.",
            "Bea gives 4 to Ada.",
            "Ada gives 2 to Bea.",
            "Bea gives 1 to Ada.",
            "Ada gives 3 to Bea.",
            "Bea gives 2 to Ada.",
            "Ada gives 1 to Bea.",
        ],
        ["8", "5"],
    ),
]


def first_line(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = t.replace("*", "").strip()
    for ln in t.splitlines():
        s = ln.strip()
        if s:
            return s
    return ""


_WALK = re.compile(
    r"\b(step|swap|initially|start(?:s|ed)? with|adding|taking|gives|moved|based on)\b",
    re.I,
)


def lead_nums(text: str) -> list[str]:
    line = first_line(text)
    if not line or _WALK.search(line):
        return []
    return re.findall(r"(?<!\d)(\d+)(?!\d)", line)


def assess(text: str, item: dict) -> dict:
    expect = list(item.get("expect") or [])
    nums = lead_nums(text)
    if item.get("family") == "give":
        wrote = nums[: len(expect)] == expect
    else:
        wrote = bool(nums) and nums[0] == expect[0]
    return {
        "empty": not (text or "").strip(),
        "n_chars": len(text or ""),
        "lead": nums,
        "wrote_result": wrote,
        "preview": (text or "").strip()[:240],
    }


def wrong_live(item: dict, live_txt: str) -> str:
    expect = item["expect"][0]
    wrong = str(item["wrong"])
    if expect in live_txt:
        return live_txt.replace(expect, wrong, 1)
    return f"the live value is {wrong}"


def mine_exp(rows: list[dict]) -> dict:
    n = len(rows)
    alone = sum(1 for r in rows if r["alone"]["wrote_result"])
    file_n = sum(1 for r in rows if r["file"]["wrote_result"])
    gate_n = sum(1 for r in rows if r["gate"]["wrote_result"])
    by_n: dict[str, dict[str, int]] = {}
    for r in rows:
        key = f"{r['family']}_n{r['n']}"
        slot = by_n.setdefault(key, {"alone": 0, "file": 0, "gate": 0})
        if r["alone"]["wrote_result"]:
            slot["alone"] += 1
        if r["file"]["wrote_result"]:
            slot["file"] += 1
        if r["gate"]["wrote_result"]:
            slot["gate"] += 1
    persist = file_n > alone or gate_n > alone
    return {
        "n": n,
        "alone_correct": alone,
        "file_correct": file_n,
        "gate_correct": gate_n,
        "by_item": by_n,
        "reasoning_persists": bool(persist),
        "findings": [
            f"Lead-correct: alone {alone}/{n}, file {file_n}/{n}, gate {gate_n}/{n}.",
            "Raise persists at higher count." if persist else "Raise did not persist at higher count.",
        ],
    }


def mine_ablate(rows: list[dict]) -> dict:
    keys = ["file_stale", "file_wrong", "file_empty", "gate_wrong"]
    n = len(rows)
    counts = {k: sum(1 for r in rows if (r.get(k) or {}).get("wrote_result")) for k in keys}
    live = sum(1 for r in rows if (r.get("file") or {}).get("wrote_result"))
    findings = [
        f"Ablation lead-correct ({n} items): "
        + ", ".join(f"{k} {counts[k]}/{n}" for k in keys)
        + f", live-file {live}/{n}."
    ]
    if counts["file_stale"] >= live and live:
        findings.append("Stale file matched live file — raise is not last-update persistence.")
    elif live > counts["file_stale"] and live > counts["file_wrong"]:
        findings.append("Only the current file raised the mouth.")
    else:
        findings.append("Ablation did not isolate the current file.")
    return {"n": n, "counts": counts, "live_file": live, "findings": findings}


def _token_ids(bundle, surfaces: list[str]) -> list[int]:
    from llmintent.predictbind import lookup_token_id

    out: list[int] = []
    for s in surfaces:
        i = lookup_token_id(bundle.tokenizer, s)
        if i is not None and i not in out:
            out.append(i)
    return out


def _gen_file(bundle, item: dict, live_txt: str, max_new: int) -> str:
    from llmintent.forward import encode_prompt_ids
    from llmintent.predictbind import greedy_generate

    prompt = f"{item['text']}\n\nLive entity state: {live_txt}."
    ids = encode_prompt_ids(bundle, prompt, thinking=False)
    return greedy_generate(bundle, ids, max_new_tokens=max_new, mode="alone")


def _gen_gate(bundle, item: dict, live: list[str], rivals: list[str], max_new: int, *, swap: bool) -> str:
    from llmintent.forward import encode_prompt_ids
    from llmintent.predictbind import greedy_generate

    ids = encode_prompt_ids(bundle, item["text"], thinking=False)
    live_ids = _token_ids(bundle, live)
    rival_ids = [i for i in _token_ids(bundle, rivals) if i not in live_ids]
    if swap:
        live_ids, rival_ids = rival_ids, live_ids
    return greedy_generate(
        bundle,
        ids,
        max_new_tokens=max_new,
        mode="alone",
        state_gate=True,
        live_ids=live_ids,
        rival_ids=rival_ids,
    )


def run_experiment(bundle, *, max_new: int, out: Path) -> dict:
    from llmintent.forward import encode_prompt_ids
    from llmintent.predictbind import greedy_generate
    from llmintent.worldfile import apply_world

    rows: list[dict] = []
    for item in ITEMS:
        world = apply_world(item["text"])
        live = world.value_surfaces()
        rivals = world.rival_surfaces()
        live_txt = world.format_live()
        print(f"{item['id']} live={live} rivals={rivals} file='{live_txt}'", file=sys.stderr, flush=True)
        ids = encode_prompt_ids(bundle, item["text"], thinking=False)
        print(f"{item['id']} gen-alone", file=sys.stderr, flush=True)
        alone_txt = greedy_generate(bundle, ids, max_new_tokens=max_new, mode="alone")
        print(f"{item['id']} gen-file", file=sys.stderr, flush=True)
        file_txt = _gen_file(bundle, item, live_txt, max_new)
        print(f"{item['id']} gen-gate", file=sys.stderr, flush=True)
        gate_txt = _gen_gate(bundle, item, live, rivals, max_new, swap=False)
        row = {
            "id": item["id"],
            "family": item["family"],
            "n": item["n"],
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
        summary = mine_exp(rows)
        out.write_text(json.dumps({"partial": True, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(
            f"{item['id']} alone={row['alone']['wrote_result']} "
            f"file={row['file']['wrote_result']} gate={row['gate']['wrote_result']} "
            f"leads={row['alone']['lead']}/{row['file']['lead']}/{row['gate']['lead']}",
            file=sys.stderr,
            flush=True,
        )
    summary = mine_exp(rows)
    payload = {
        "model": getattr(bundle, "name", "qwen:27b"),
        "site": "count_scale_experiment",
        "partial": False,
        "summary": summary,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return payload


def run_ablation(bundle, *, max_new: int, exp: dict, out: Path) -> dict:
    from llmintent.worldfile import apply_world

    by_id = {r["id"]: r for r in exp.get("rows") or []}
    rows: list[dict] = []
    for item in ITEMS:
        world = apply_world(item["text"])
        stale_w = apply_world(item["stale_text"])
        live = world.value_surfaces()
        rivals = world.rival_surfaces()
        live_txt = world.format_live()
        stale_txt = stale_w.format_live()
        wrong_txt = wrong_live(item, live_txt)
        exp_row = by_id.get(item["id"]) or {}
        print(f"{item['id']} ablate stale='{stale_txt}' wrong='{wrong_txt}'", file=sys.stderr, flush=True)
        print(f"{item['id']} gen-file_stale", file=sys.stderr, flush=True)
        stale_out = _gen_file(bundle, item, stale_txt, max_new)
        print(f"{item['id']} gen-file_wrong", file=sys.stderr, flush=True)
        wrong_out = _gen_file(bundle, item, wrong_txt, max_new)
        print(f"{item['id']} gen-file_empty", file=sys.stderr, flush=True)
        empty_out = _gen_file(bundle, item, "(none)", max_new)
        print(f"{item['id']} gen-gate_wrong", file=sys.stderr, flush=True)
        gwrong = _gen_gate(bundle, item, live, rivals, max_new, swap=True)
        row = {
            "id": item["id"],
            "family": item["family"],
            "n": item["n"],
            "expect": item["expect"],
            "format_live": live_txt,
            "format_stale": stale_txt,
            "format_wrong": wrong_txt,
            "file_text": exp_row.get("file_text"),
            "file": exp_row.get("file") or assess(exp_row.get("file_text") or "", item),
            "file_stale_text": stale_out,
            "file_wrong_text": wrong_out,
            "file_empty_text": empty_out,
            "gate_wrong_text": gwrong,
            "file_stale": assess(stale_out, item),
            "file_wrong": assess(wrong_out, item),
            "file_empty": assess(empty_out, item),
            "gate_wrong": assess(gwrong, item),
        }
        rows.append(row)
        out.write_text(
            json.dumps({"partial": True, "summary": mine_ablate(rows), "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(
            f"{item['id']} stale={row['file_stale']['wrote_result']} "
            f"wrong={row['file_wrong']['wrote_result']} "
            f"empty={row['file_empty']['wrote_result']} "
            f"gwrong={row['gate_wrong']['wrote_result']}",
            file=sys.stderr,
            flush=True,
        )
    summary = mine_ablate(rows)
    payload = {
        "model": getattr(bundle, "name", "qwen:27b"),
        "site": "count_scale_ablation",
        "partial": False,
        "summary": summary,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return payload


def load_model(model: str, load_in_4bit: bool | None):
    from llmintent.suite import load_suite_model, resolve_model_spec

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(f"loading {model} 4bit={bool(four)} count-persist", file=sys.stderr, flush=True)
    return load_suite_model(model=model, load_in_4bit=four)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen:27b")
    p.add_argument("--4bit", dest="load_in_4bit", action="store_true")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--phase", choices=("experiment", "ablation", "both"), default="both")
    p.add_argument("--exp-output", default="artifacts/count_persist_27b.json")
    p.add_argument("--ablate-output", default="artifacts/count_ablate_27b.json")
    args = p.parse_args(argv)
    exp_path = Path(args.exp_output)
    ab_path = Path(args.ablate_output)
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    four = True if args.load_in_4bit else None
    bundle = None
    exp = None
    if args.phase in ("experiment", "both"):
        bundle = load_model(args.model, four)
        exp = run_experiment(bundle, max_new=args.max_new, out=exp_path)
    if args.phase in ("ablation", "both"):
        if exp is None:
            if not exp_path.exists():
                raise SystemExit(f"need {exp_path} before ablation")
            exp = json.loads(exp_path.read_text(encoding="utf-8"))
        if bundle is None:
            bundle = load_model(args.model, four)
        run_ablation(bundle, max_new=args.max_new, exp=exp, out=ab_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
