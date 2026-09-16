"""100-prompt numerical-pole ablation: mine how intensity behaves.

Runs GPT-2 (or --model). Each base prompt gets five texts matching the
classic sweep: original, insert 5, change number to 10, drop a content
noun, swap a content noun. Intensity is last-token cosine vs the
numerical pole. Compile occupancy is recorded as a cheap second density.

This mines correlates. It does not identify a number circuit.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "twelve": "12",
    "twenty": "20",
    "twenty-five": "25",
}
_NUM = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|twelve|"
    r"twenty-five|twenty|\d+)\b",
    re.I,
)
_NOUN = re.compile(
    r"\b(fox|dog|cat|bird|spider|horse|cow|sheep|apple|egg|box|river|"
    r"valley|girl|boy|car|house|tree|book|hat|mat|fish|bear|wolf)\b",
    re.I,
)


@dataclass
class Item:
    id: int
    family: str
    original: str
    noun: str
    has_number: bool


def _n(word: str, n: int) -> str:
    return f"{word} {n}" if n != 1 else word


def build_corpus() -> list[Item]:
    items: list[Item] = []
    n = 0

    arith = [
        (8, "minus", 2),
        (12, "plus", 4),
        (10, "minus", 3),
        (7, "plus", 5),
        (15, "minus", 6),
        (9, "times", 2),
        (20, "minus", 8),
        (6, "plus", 6),
        (14, "minus", 7),
        (11, "plus", 9),
        (4, "times", 3),
        (18, "minus", 9),
        (13, "plus", 2),
        (16, "minus", 4),
        (5, "times", 5),
        (21, "minus", 7),
        (3, "plus", 8),
        (30, "minus", 12),
        (2, "times", 9),
        (17, "plus", 3),
        (25, "minus", 5),
        (8, "plus", 7),
        (19, "minus", 4),
        (6, "times", 4),
        (22, "minus", 11),
    ]
    for a, op, b in arith:
        items.append(
            Item(
                n,
                "arithmetic",
                f"{a} {op} {b} equals",
                "equals",
                True,
            )
        )
        n += 1

    counts = [
        ("spider", 8, "legs"),
        ("carton", 12, "eggs"),
        ("week", 7, "days"),
        ("clock", 12, "hours"),
        ("hand", 5, "fingers"),
        ("car", 4, "wheels"),
        ("year", 12, "months"),
        ("box", 6, "apples"),
        ("team", 11, "players"),
        ("piano", 88, "keys"),
        ("dog", 4, "legs"),
        ("bird", 2, "wings"),
        ("cow", 4, "stomachs"),
        ("tree", 3, "branches"),
        ("house", 8, "windows"),
        ("book", 10, "chapters"),
        ("hat", 2, "sides"),
        ("fish", 7, "stripes"),
        ("bear", 2, "ears"),
        ("wolf", 4, "paws"),
        ("horse", 4, "hooves"),
        ("sheep", 2, "horns"),
        ("cat", 9, "lives"),
        ("fox", 4, "paws"),
        ("girl", 3, "books"),
    ]
    for noun, k, thing in counts:
        items.append(
            Item(n, "counting", f"The {noun} has {k} {thing}.", noun, True)
        )
        n += 1

    stories = [
        ("fox", "jumps over", "dog", 3),
        ("cat", "sat on", "mat", 2),
        ("bird", "flew over", "tree", 5),
        ("dog", "chased", "cat", 1),
        ("horse", "crossed", "river", 4),
        ("cow", "ate", "apple", 6),
        ("sheep", "stood by", "house", 2),
        ("wolf", "watched", "sheep", 8),
        ("bear", "climbed", "tree", 1),
        ("fish", "swam under", "boat", 7),
        ("boy", "opened", "box", 3),
        ("girl", "read", "book", 4),
        ("cat", "wore", "hat", 1),
        ("fox", "hid behind", "tree", 2),
        ("dog", "slept on", "mat", 5),
        ("bird", "built", "house", 3),
        ("horse", "kicked", "box", 2),
        ("cow", "walked into", "valley", 1),
        ("wolf", "howled at", "house", 4),
        ("bear", "found", "fish", 6),
        ("spider", "spun over", "hat", 8),
        ("cat", "knocked", "book", 2),
        ("fox", "stole", "egg", 3),
        ("dog", "buried", "bone", 1),
        ("girl", "carried", "apple", 5),
    ]
    for a, verb, b, k in stories:
        items.append(
            Item(
                n,
                "story",
                f"The {a} {verb} {k} {b}s." if k != 1 else f"The {a} {verb} {k} {b}.",
                a,
                True,
            )
        )
        n += 1

    prose = [
        "The fox jumps over the lazy dog.",
        "The cat sat on the mat.",
        "A river runs through the valley.",
        "The girl lived in a quiet house.",
        "The bird sang in the tree.",
        "A horse walked beside the river.",
        "The dog slept near the fire.",
        "The wolf watched the sheep.",
        "A bear climbed the old tree.",
        "The fish swam under the boat.",
        "The boy opened the wooden box.",
        "She read the book by the window.",
        "The cat wore a small hat.",
        "A fox hid behind the tree.",
        "The cow stood in the valley.",
        "The sheep followed the girl.",
        "A car sat by the house.",
        "The hat fell on the mat.",
        "The tree shaded the house.",
        "A book lay on the mat.",
        "The river met the valley.",
        "The girl called the dog.",
        "A bird left the tree.",
        "The wolf crossed the river.",
        "The bear sniffed the box.",
    ]
    for text in prose:
        m = _NOUN.search(text)
        items.append(Item(n, "prose", text, (m.group(1) if m else "fox").lower(), False))
        n += 1

    if len(items) != 100:
        raise RuntimeError(f"expected 100 bases, got {len(items)}")
    return items


def _swap_noun(text: str, noun: str) -> str:
    alt = "cat" if noun.lower() != "cat" else "dog"
    pat = re.compile(rf"\b{re.escape(noun)}\b", re.I)
    if pat.search(text):
        return pat.sub(alt, text, count=1)
    m = _NOUN.search(text)
    if m:
        other = "dog" if m.group(1).lower() != "dog" else "cat"
        return text[: m.start()] + other + text[m.end() :]
    return text.replace("the", "a cat in the", 1)


def _drop_noun(text: str, noun: str) -> str:
    pat = re.compile(rf"\b{re.escape(noun)}\b\s*", re.I)
    out = pat.sub("", text, count=1)
    return re.sub(r"\s+", " ", out).strip() or text


def _insert_five(text: str) -> str:
    if _NUM.search(text):
        return _NUM.sub("5", text, count=1)
    m = _NOUN.search(text)
    if m:
        return text[: m.start()] + "5 " + text[m.start() :]
    return "5 " + text


def _to_ten(text: str) -> str:
    if _NUM.search(text):
        return _NUM.sub("10", text, count=1)
    m = _NOUN.search(text)
    if m:
        return text[: m.start()] + "10 " + text[m.start() :]
    return "10 " + text


def conditions_for(item: Item) -> dict[str, str]:
    return {
        "original": item.original,
        "with_number_5": _insert_five(item.original),
        "changed_number_10": _to_ten(item.original),
        "drop_noun": _drop_noun(item.original, item.noun),
        "swap_noun": _swap_noun(item.original, item.noun),
    }


def _bands(n: int) -> dict[str, list[int]]:
    early = list(range(0, max(1, int(0.34 * n))))
    late = list(range(int(0.72 * n), n))
    mid = [i for i in range(n) if i not in early and i not in late]
    return {"early": early, "mid": mid, "late": late}


def _band_mean(curve: list[float], idx: list[int]) -> float:
    if not idx:
        return 0.0
    return float(mean(curve[i] for i in idx))


def mine(rows: list[dict], n_layers: int) -> dict:
    bands = _bands(n_layers)
    by_cond: dict[str, list[list[float]]] = {}
    paired: dict[str, list[float]] = {}
    originals = {r["id"]: r for r in rows if r["condition"] == "original"}

    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r["curve"])
        if r["condition"] == "original":
            continue
        base = originals.get(r["id"])
        if not base:
            continue
        d_late = _band_mean(r["curve"], bands["late"]) - _band_mean(
            base["curve"], bands["late"]
        )
        paired.setdefault(r["condition"], []).append(d_late)

    layer_means = {
        cond: [float(mean(curves[i][j] for i in range(len(curves)))) for j in range(n_layers)]
        for cond, curves in by_cond.items()
    }

    def sign_share(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean_delta_late": 0.0, "frac_up": 0.0, "frac_down": 0.0}
        up = sum(1 for v in vals if v > 1e-4)
        down = sum(1 for v in vals if v < -1e-4)
        return {
            "n": len(vals),
            "mean_delta_late": round(mean(vals), 5),
            "median_delta_late": round(median(vals), 5),
            "frac_up": round(up / len(vals), 3),
            "frac_down": round(down / len(vals), 3),
        }

    # Digit presence vs late intensity on originals only.
    orig = [r for r in rows if r["condition"] == "original"]
    with_d = [r for r in orig if r["has_number"]]
    no_d = [r for r in orig if not r["has_number"]]
    digit = {
        "late_mean_with_number": round(
            mean(_band_mean(r["curve"], bands["late"]) for r in with_d), 5
        )
        if with_d
        else None,
        "late_mean_no_number": round(
            mean(_band_mean(r["curve"], bands["late"]) for r in no_d), 5
        )
        if no_d
        else None,
        "early_mean_with_number": round(
            mean(_band_mean(r["curve"], bands["early"]) for r in with_d), 5
        )
        if with_d
        else None,
        "early_mean_no_number": round(
            mean(_band_mean(r["curve"], bands["early"]) for r in no_d), 5
        )
        if no_d
        else None,
    }

    family_late: dict[str, float] = {}
    for fam in sorted({r["family"] for r in orig}):
        fam_rows = [r for r in orig if r["family"] == fam]
        family_late[fam] = round(
            mean(_band_mean(r["curve"], bands["late"]) for r in fam_rows), 5
        )

    # Compile: does causal_logic occupancy track late numerical intensity?
    compile_pairs = [
        (float(r["compile"].get("causal_logic", 0.0)), _band_mean(r["curve"], bands["late"]))
        for r in orig
    ]
    if len(compile_pairs) > 2:
        mx = mean(a for a, _ in compile_pairs)
        my = mean(b for _, b in compile_pairs)
        num = sum((a - mx) * (b - my) for a, b in compile_pairs)
        den = (
            sum((a - mx) ** 2 for a, _ in compile_pairs) ** 0.5
            * sum((b - my) ** 2 for _, b in compile_pairs) ** 0.5
        )
        corr = float(num / den) if den else 0.0
    else:
        corr = 0.0

    findings = []
    d5 = sign_share(paired.get("with_number_5", []))
    d10 = sign_share(paired.get("changed_number_10", []))
    drop = sign_share(paired.get("drop_noun", []))
    swap = sign_share(paired.get("swap_noun", []))

    if d5["mean_delta_late"] <= 0 and d10["mean_delta_late"] <= 0:
        findings.append(
            "Inserting or changing a number did not raise late numerical-pole intensity "
            "on average. The pole is not tracking 'there is a digit in the prompt'."
        )
    else:
        findings.append(
            "A number edit raised late intensity on average — weak support that the "
            "pole can follow surface digits (still a correlate)."
        )
    findings.append(
        f"Dropping a content noun shifted late intensity {drop['mean_delta_late']:+.4f} "
        f"(up on {drop['frac_up']:.0%}, down on {drop['frac_down']:.0%})."
    )
    findings.append(
        f"Swapping a noun shifted late intensity {swap['mean_delta_late']:+.4f} "
        f"(up on {swap['frac_up']:.0%}, down on {swap['frac_down']:.0%}). "
        "A non-number edit moving the score means the probe is leaky."
    )
    if digit["late_mean_with_number"] is not None and digit["late_mean_no_number"] is not None:
        gap = digit["late_mean_with_number"] - digit["late_mean_no_number"]
        findings.append(
            f"Originals that already contain a number vs those that do not: late-band "
            f"gap {gap:+.4f} (with={digit['late_mean_with_number']}, "
            f"without={digit['late_mean_no_number']})."
        )
    findings.append(
        f"Compile causal_logic occupancy vs late intensity correlation r={corr:.3f} "
        "(near zero means the atlas density and the number pole are not the same thing)."
    )

    return {
        "n_layers": n_layers,
        "n_bases": 100,
        "n_forwards": len(rows),
        "bands": {k: [min(v), max(v)] if v else [] for k, v in bands.items()},
        "layer_means": layer_means,
        "paired_late": {
            "with_number_5": d5,
            "changed_number_10": d10,
            "drop_noun": drop,
            "swap_noun": swap,
        },
        "digit_split": digit,
        "family_late_original": family_late,
        "compile_causal_vs_late_r": round(corr, 4),
        "findings": findings,
    }


def run(
    *,
    model: str = "gpt2",
    limit: int | None = None,
    load_in_4bit: bool | None = None,
    checkpoint: Path | None = None,
) -> dict:
    import sys

    from llmintent.anatomy.compile import compile_regions
    from llmintent.forward import forward_hidden_states
    from llmintent.metrics import cosine_intensity
    from llmintent.poles import build_numerical_pole
    from llmintent.suite import load_suite_model, resolve_model_spec

    corpus = build_corpus()
    if limit:
        corpus = corpus[: int(limit)]
    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(
        f"loading {model} 4bit={bool(four)}  bases={len(corpus)} forwards={len(corpus) * 5}",
        file=sys.stderr,
        flush=True,
    )
    bundle = load_suite_model(model=model, load_in_4bit=four)
    try:
        pole = build_numerical_pole(bundle)
    except ValueError:
        pole = build_numerical_pole(
            bundle,
            tokens=["0", "1", "2", "3", "4", "5", " 0", " 1", "5", "10"],
        )
    rows: list[dict] = []
    total = len(corpus) * 5
    done = 0

    for item in corpus:
        conds = conditions_for(item)
        for cond, text in conds.items():
            _, states = forward_hidden_states(bundle, text)
            h0 = states[0][0, -1, :]
            pole_d = pole.to(device=h0.device, dtype=h0.dtype)
            curve = [
                cosine_intensity(states[i][0, -1, :], pole_d) for i in range(len(states))
            ]
            plan = compile_regions(text)
            rows.append(
                {
                    "id": item.id,
                    "family": item.family,
                    "condition": cond,
                    "text": text,
                    "has_number": bool(_NUM.search(text)),
                    "curve": [round(x, 6) for x in curve],
                    "peak_layer": int(max(range(len(curve)), key=lambda i: curve[i])),
                    "peak": round(max(curve), 6),
                    "compile": {k: round(v, 4) for k, v in plan.occupancy().items() if v > 0},
                }
            )
            done += 1
            if done % 10 == 0 or done == total:
                print(f"forward {done}/{total}", file=sys.stderr, flush=True)
                if checkpoint is not None:
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    n = len(rows[0]["curve"]) if rows else 0
                    payload = {
                        "summary": mine(rows, n) if n else {},
                        "rows": rows,
                        "partial": done < total,
                    }
                    payload["summary"]["model"] = getattr(bundle, "name", model)
                    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    summary = mine(rows, len(rows[0]["curve"]) if rows else int(bundle.num_layers))
    summary["model"] = getattr(bundle, "name", model)
    summary["load_in_4bit"] = bool(four)
    return {"summary": summary, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="100-prompt numerical pole ablation")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--4bit", dest="load_in_4bit", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Use first N bases (debug)")
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args(argv)

    out = Path(args.output) if args.output else Path("artifacts") / "pole_ablation_100.json"
    payload = run(
        model=args.model,
        limit=args.limit,
        load_in_4bit=True if args.load_in_4bit else None,
        checkpoint=out,
    )
    text = json.dumps(payload["summary"], indent=2)
    print(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["partial"] = False
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
