"""Hidden-number-box test on a causal LM (default: Qwen 27B NF4).

Raw tokenization (no chat wrapper) so the last token is the pinned word.
One forward per text. Next-token argmax is the written first piece.
Quantity probe: last residual vs result-digit embedding, vs a wrong digit,
vs 'cat'. Causal: inject (embed(result) - mean digit embeds) at mid layers.

Does not claim a fly neuropil. Kill the box if last-token identity returns,
if the written digit is chance, or if random steer matches result steer.
"""

from __future__ import annotations

import argparse
import json
import random
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
    core = f"{a} {op} {b}"
    return {
        "pinned": f"{core}. The answer is",
        "equals": f"{core} equals",
        "cat": f"{core}. The cat is",
    }


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


def digit_token_ids(bundle) -> dict[int, int]:
    out: dict[int, int] = {}
    for n in range(0, 31):
        tid = _tok_id(bundle, str(n)) or _tok_id(bundle, f" {n}")
        if tid is not None:
            out[n] = tid
    return out


def last_hidden(bundle, text: str):
    from llmintent.forward import forward_hidden_states_from_ids, get_lm_head, normalize_hidden

    ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
    last_id = int(ids[0, -1].item())
    last_str = bundle.tokenizer.decode([last_id])
    states = forward_hidden_states_from_ids(bundle, ids)
    h = normalize_hidden(bundle, states[-1][0, -1, :].float())
    h0 = states[0][0, -1, :].float()
    head = get_lm_head(bundle)
    try:
        logits = head(h.to(dtype=next(head.parameters()).dtype)).float()
    except Exception:
        from llmintent.models import get_unembedding_matrix

        unembed = get_unembedding_matrix(bundle.model).float()
        logits = F.linear(h.cpu(), unembed.cpu()).float().to(h.device)
    return {
        "ids": ids,
        "last_id": last_id,
        "last_str": last_str,
        "h": h,
        "h0": h0,
        "logits": logits,
        "n_states": len(states),
    }


def embed_row(bundle, tid: int) -> torch.Tensor:
    from llmintent.models import get_input_embeddings

    E = get_input_embeddings(bundle.model)
    return E[tid].float().detach()


@torch.no_grad()
def steered_top(bundle, text: str, vec: torch.Tensor, layers: list[int], gain: float) -> str:
    from llmintent.forward import (
        forward_hidden_states_from_ids,
        get_lm_head,
        normalize_hidden,
    )
    from llmintent.heighten.intervention import steering_hooks

    ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
    with steering_hooks(bundle, layers, vec, gain):
        states = forward_hidden_states_from_ids(bundle, ids)
    head = get_lm_head(bundle)
    last = normalize_hidden(bundle, states[-1][0, -1, :].float())
    try:
        logits = head(last.to(dtype=next(head.parameters()).dtype)).float()
    except Exception:
        from llmintent.models import get_unembedding_matrix

        unembed = get_unembedding_matrix(bundle.model).float()
        logits = F.linear(last.cpu(), unembed.cpu()).float()
    top = int(torch.argmax(logits).item())
    return bundle.tokenizer.decode([top], skip_special_tokens=True).strip() or f"id:{top}"


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def mine(rows: list[dict]) -> dict:
    pinned = [r for r in rows if r["ending"] == "pinned"]
    n = len(pinned) or 1
    written = sum(1 for r in pinned if r["written_correct"])
    box_pref = []
    for r in pinned:
        box_pref.append(1 if r["cos_result"] > r["cos_wrong"] and r["cos_result"] > r["cos_cat"] else 0)
    ident_delta = []
    cat_delta = []
    by_item: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_item.setdefault(r["i"], {})[r["ending"]] = r
    for item in by_item.values():
        if "pinned" in item and "equals" in item:
            ident_delta.append(item["equals"]["p_result"] - item["pinned"]["p_result"])
        if "pinned" in item and "cat" in item:
            cat_delta.append(item["cat"]["p_result"] - item["pinned"]["p_result"])
    last_ok = sum(1 for r in pinned if r["last_str"].strip().lower() in {"is", "is "})
    findings = []
    wr = written / n
    findings.append(
        f"Wrote the correct next-token on {written}/{len(pinned)} pinned prompts ({wr:.0%})."
    )
    pref = sum(box_pref) / n
    findings.append(
        f"Last residual closer to the result digit than to a wrong digit and 'cat' "
        f"on {sum(box_pref)}/{len(pinned)} ({pref:.0%}). Chance if no box is ~1/3."
    )
    if ident_delta:
        m = float(np.mean(ident_delta))
        findings.append(
            f"Switching the last word to 'equals' changed P(result) by {m:+.3f} on average. "
            "A real box should stay put; identity follows the last word."
        )
    if cat_delta:
        m = float(np.mean(cat_delta))
        findings.append(
            f"Switching the last word to 'cat is' changed P(result) by {m:+.3f} on average."
        )
    steer = [r for r in rows if r.get("steer")]
    if steer:
        flip_res = sum(1 for r in steer if r["steer"]["result_top"] != r["steer"]["base_top"])
        flip_rnd = sum(1 for r in steer if r["steer"]["random_top"] != r["steer"]["base_top"])
        findings.append(
            f"Result-direction steer flipped the next token on {flip_res}/{len(steer)}; "
            f"random steer flipped {flip_rnd}/{len(steer)}."
        )
        if flip_res <= flip_rnd:
            findings.append(
                "Steer did not beat random. No causal number box in this injection."
            )
    if wr < 0.25:
        findings.append(
            "Generation of the digit is weak. A hidden box is a different claim from "
            "being able to finish the equation."
        )
    if pref <= 0.4 and wr >= 0.25:
        findings.append(
            "It can write the answer without the last residual pointing at the result "
            "digit. Next-word knowledge, not a readable box."
        )
    return {
        "n_pinned": len(pinned),
        "written_correct_frac": round(wr, 3),
        "box_preference_frac": round(pref, 3),
        "last_token_is_frac": round(last_ok / n, 3),
        "mean_p_result_pinned": round(float(np.mean([r["p_result"] for r in pinned])), 5)
        if pinned
        else 0.0,
        "mean_ident_delta_p": round(float(np.mean(ident_delta)), 5) if ident_delta else None,
        "mean_cat_delta_p": round(float(np.mean(cat_delta)), 5) if cat_delta else None,
        "mean_cos_result": round(float(np.mean([r["cos_result"] for r in pinned])), 4)
        if pinned
        else 0.0,
        "mean_cos_wrong": round(float(np.mean([r["cos_wrong"] for r in pinned])), 4)
        if pinned
        else 0.0,
        "mean_cos_cat": round(float(np.mean([r["cos_cat"] for r in pinned])), 4) if pinned else 0.0,
        "findings": findings,
    }


def run(*, model: str, out: Path, load_in_4bit: bool | None, steer_n: int, gain: float) -> dict:
    from llmintent.forward import get_lm_head
    from llmintent.suite import load_suite_model, resolve_model_spec

    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(f"loading {model} 4bit={bool(four)} items={len(ITEMS)}", file=sys.stderr, flush=True)
    bundle = load_suite_model(model=model, load_in_4bit=four)
    digits = digit_token_ids(bundle)
    cat_id = _tok_id(bundle, "cat") or _tok_id(bundle, " cat")
    if cat_id is None:
        cat_id = int(bundle.tokenizer.encode("cat", add_special_tokens=False)[0])

    digit_vecs = [embed_row(bundle, tid) for tid in digits.values()]
    mean_digit = torch.stack(digit_vecs).mean(0)
    n_layers = int(bundle.num_layers)
    mid = [n_layers // 2, (n_layers * 2) // 3]

    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        rows = prev.get("rows") or []
        done = {(r["i"], r["ending"]) for r in rows}
        print(f"resume {len(rows)} rows", file=sys.stderr, flush=True)
    else:
        done = set()

    rng = random.Random(0)
    for i, (a, op, b, result) in enumerate(ITEMS):
        rid, rpiece = result_token_id(bundle, result)
        wrong_n = result + 1 if result + 1 <= 30 else result - 1
        wid = digits.get(wrong_n) or rid
        e_res, e_wrong, e_cat = embed_row(bundle, rid), embed_row(bundle, wid), embed_row(bundle, cat_id)
        for ending, text in endings_for(a, op, b).items():
            if (i, ending) in done:
                print(f"skip {i} {ending}", file=sys.stderr, flush=True)
                continue
            pack = last_hidden(bundle, text)
            logits = pack["logits"]
            probs = F.softmax(logits, dim=-1)
            p_res = float(probs[rid].item())
            top_id = int(torch.argmax(logits).item())
            top = bundle.tokenizer.decode([top_id], skip_special_tokens=True)
            written_correct = top_id == rid or str(result) in (top or "")
            h = pack["h"].cpu()
            row = {
                "i": i,
                "text": text,
                "ending": ending,
                "a": a,
                "op": op,
                "b": b,
                "result": result,
                "result_piece": rpiece,
                "last_id": pack["last_id"],
                "last_str": pack["last_str"],
                "top": top.strip(),
                "written_correct": bool(written_correct),
                "p_result": round(p_res, 6),
                "cos_result": round(cosine(h, e_res.cpu()), 4),
                "cos_wrong": round(cosine(h, e_wrong.cpu()), 4),
                "cos_cat": round(cosine(h, e_cat.cpu()), 4),
                "cos_mean_digits": round(cosine(h, mean_digit.cpu()), 4),
            }
            if ending == "pinned" and i < steer_n:
                vec = (e_res - mean_digit).cpu()
                vec = vec / (vec.norm() + 1e-8)
                rnd = torch.randn_like(vec)
                rnd = rnd / (rnd.norm() + 1e-8)
                base_top = (top or "").strip()
                try:
                    res_top = steered_top(bundle, text, vec.to(bundle.device), mid, gain)
                    rnd_top = steered_top(bundle, text, rnd.to(bundle.device), mid, gain)
                    row["steer"] = {
                        "base_top": base_top,
                        "result_top": res_top,
                        "random_top": rnd_top,
                        "layers": mid,
                        "gain": gain,
                    }
                except Exception as exc:
                    row["steer"] = {"error": str(exc), "base_top": base_top}
            rows.append(row)
            done.add((i, ending))
            summary = mine(rows)
            payload = {
                "model": getattr(bundle, "name", model),
                "partial": True,
                "summary": summary,
                "rows": rows,
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                f"wrote {i} {ending} last={pack['last_str']!r} top={top!r} "
                f"p={p_res:.4f} correct={written_correct}",
                file=sys.stderr,
                flush=True,
            )

    summary = mine(rows)
    payload = {
        "model": getattr(bundle, "name", model),
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
    p.add_argument("-o", "--output", default="artifacts/number_box_27b.json")
    p.add_argument("--steer-n", type=int, default=4)
    p.add_argument("--gain", type=float, default=2.0)
    args = p.parse_args(argv)
    run(
        model=args.model,
        out=Path(args.output),
        load_in_4bit=True if args.load_in_4bit else None,
        steer_n=args.steer_n,
        gain=args.gain,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
