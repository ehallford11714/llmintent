"""Bind locked state into next-token prediction.

The residual-stream write failed: last-token paint and in-stream hooks
did not move the mouth. The gap is the readout.

Default path is a Bayesian update, not a command:

  p_post(w) ∝ p_model(w) · π_bind(w)^λ
  λ = max_strength · H(p_model) / log V

π_bind is almost uniform, with a small extra mass on locked names (and
optionally softmax(W @ g)). When the model is sure, H is low and the
bind is silent. When it is unsure, the lock informs. ``members`` /
``unembed`` / ``both`` still *direct* (+gain). Those stuttered 27B.

Residual add is not prediction. Decay is not trained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


def member_surface(name: str) -> str:
    """``digit:8`` -> ``8``. Bare names pass through."""
    if ":" in name:
        return name.split(":", 1)[1]
    return name


def lookup_token_id(tokenizer: Any, piece: str) -> int | None:
    """Single-token surfaces only. Do not take the first piece of ``12`` / ``20``."""
    if not piece:
        return None
    for cand in (piece, f" {piece}", f"{piece}."):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def member_token_ids(bundle: Any, members: Sequence[str]) -> list[int]:
    """Vocab ids for locked member names. First unique match only."""
    seen: list[int] = []
    tok = bundle.tokenizer
    for name in members:
        tid = lookup_token_id(tok, member_surface(name))
        if tid is not None and tid not in seen:
            seen.append(tid)
    return seen


def transformer_logits(bundle: Any, hidden: torch.Tensor) -> torch.Tensor:
    """Vanilla next-token scores. This is the gap if used alone."""
    from llmintent.forward import get_lm_head, normalize_hidden

    h = normalize_hidden(bundle, hidden.float())
    head = get_lm_head(bundle)
    try:
        return head(h.to(dtype=next(head.parameters()).dtype)).float().cpu().reshape(-1)
    except Exception:
        from llmintent.models import get_unembedding_matrix

        unembed = get_unembedding_matrix(bundle.model)
        return F.linear(h.cpu().float(), unembed.detach().float().cpu()).reshape(-1)


def bound_vocab_logits(bundle: Any, vector: torch.Tensor) -> torch.Tensor:
    """Project locked g into vocab space: W @ g. No extra residual write."""
    from llmintent.models import get_unembedding_matrix

    W = get_unembedding_matrix(bundle.model)
    v = vector.detach().reshape(-1).to(device=W.device, dtype=W.dtype)
    if v.numel() != int(W.shape[1]):
        raise ValueError(f"bound dim {v.numel()} != unembed {int(W.shape[1])}")
    return F.linear(v, W).float().cpu().reshape(-1)


def apply_unembed_bind(
    logits: torch.Tensor,
    extra: torch.Tensor,
    *,
    gain: float = 1.5,
) -> torch.Tensor:
    out = logits.float().reshape(-1).clone()
    add = extra.float().reshape(-1)
    if out.numel() != add.numel():
        raise ValueError(f"logits {out.numel()} != unembed bind {add.numel()}")
    return out + float(gain) * add


def bind_log_prior(
    vocab_size: int,
    member_ids: Iterable[int] = (),
    extra: torch.Tensor | None = None,
    *,
    member_mass: float = 0.02,
    geom_mass: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """π_bind = leftover·U + member_mass·U_M + geom_mass·softmax(W @ g).

    A prior, not a next-word command. Most mass stays uniform.
    """
    v = int(vocab_size)
    if v < 1:
        raise ValueError("vocab_size must be positive")
    m = max(0.0, float(member_mass))
    g = max(0.0, float(geom_mass))
    if extra is None:
        g = 0.0
    leftover = max(0.0, 1.0 - m - g)
    ids = [int(i) for i in member_ids if 0 <= int(i) < v]
    seen: list[int] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    pi = torch.full((v,), leftover / float(v), device=device, dtype=dtype or torch.float32)
    if seen and m > 0:
        pi[seen] = pi[seen] + m / float(len(seen))
    if extra is not None and g > 0:
        add = extra.float().reshape(-1).to(device=pi.device)
        if add.numel() != v:
            raise ValueError(f"extra {add.numel()} != vocab {v}")
        pi = pi + g * F.softmax(add, dim=-1)
    pi = pi.clamp(min=1e-12)
    pi = pi / pi.sum()
    return pi.log()


def bayes_inform(
    logits: torch.Tensor,
    log_prior: torch.Tensor,
    *,
    max_strength: float = 1.0,
) -> torch.Tensor:
    """p_post ∝ p_model · π^λ with λ entropy-scaled. Returns log-posterior scores."""
    scores = logits.float()
    prior = log_prior.float().reshape(-1).to(device=scores.device)
    if prior.numel() != scores.shape[-1]:
        raise ValueError(f"prior {prior.numel()} != logits {scores.shape[-1]}")
    log_lik = F.log_softmax(scores, dim=-1)
    ent = -(log_lik.exp() * log_lik).sum(dim=-1, keepdim=True)
    h_max = float(torch.log(torch.tensor(float(scores.shape[-1]))))
    lam = float(max_strength) * (ent / max(h_max, 1e-8)).clamp(0.0, 1.0)
    return log_lik + lam * prior


def apply_member_bind(
    logits: torch.Tensor,
    token_ids: Iterable[int],
    *,
    gain: float = 8.0,
) -> torch.Tensor:
    out = logits.float().reshape(-1).clone()
    n = out.numel()
    for tid in token_ids:
        i = int(tid)
        if 0 <= i < n:
            out[i] = out[i] + float(gain)
    return out


def predict_from_bind(
    bundle: Any,
    hidden: torch.Tensor,
    *,
    bound_vector: torch.Tensor | None = None,
    members: Sequence[str] = (),
    mode: str = "bayes",
    gain_unembed: float = 1.5,
    gain_members: float = 8.0,
    member_ids: Sequence[int] | None = None,
    member_mass: float = 0.02,
    geom_mass: float = 0.0,
    max_strength: float = 1.0,
) -> torch.Tensor:
    """Next-token scores that use the lock.

    Default ``bayes`` informs. ``members`` / ``unembed`` / ``both`` direct.
    """
    if mode not in ("unembed", "members", "both", "bayes"):
        raise ValueError(f"unknown prediction-bind mode {mode!r}")
    logits = transformer_logits(bundle, hidden)
    extra = None
    if bound_vector is not None and mode in ("unembed", "both", "bayes"):
        extra = bound_vocab_logits(bundle, bound_vector)
    ids = list(member_ids) if member_ids is not None else member_token_ids(bundle, members)
    if mode == "bayes":
        return bind_scores(
            logits,
            extra=extra,
            member_ids=ids,
            mode="bayes",
            member_mass=member_mass,
            geom_mass=geom_mass,
            max_strength=max_strength,
        )
    if mode in ("unembed", "both") and extra is not None:
        logits = apply_unembed_bind(logits, extra, gain=gain_unembed)
    if mode in ("members", "both"):
        logits = apply_member_bind(logits, ids, gain=gain_members)
    return logits


@dataclass
class PredictBindReport:
    mode: str
    top_id: int
    top_piece: str
    n_members: int
    member_ids: list[int]


def apply_state_gate(
    scores: torch.Tensor,
    live_ids: Iterable[int],
    rival_ids: Iterable[int],
    *,
    boost: float = 6.0,
) -> tuple[torch.Tensor, bool]:
    """When the next token is a live or rival state value, prefer the file.

    Fires only on a state choice (argmax in live∪rival). Walkthrough tokens
    stay vanilla. Returns (scores, fired).
    """
    out = scores.float().clone()
    live = [int(i) for i in live_ids]
    rival = [int(i) for i in rival_ids]
    watch = set(live + rival)
    if not watch:
        return out, False
    if out.dim() == 1:
        top = int(out.argmax().item())
        if top not in watch:
            return out, False
        for r in rival:
            if 0 <= r < out.numel():
                out[r] = out[r] - float(boost)
        for i in live:
            if 0 <= i < out.numel():
                out[i] = out[i] + float(boost)
        return out, True
    top = int(out.reshape(-1, out.shape[-1]).argmax(dim=-1)[0].item())
    if top not in watch:
        return out, False
    for r in rival:
        if 0 <= r < out.shape[-1]:
            out[..., r] = out[..., r] - float(boost)
    for i in live:
        if 0 <= i < out.shape[-1]:
            out[..., i] = out[..., i] + float(boost)
    return out, True


def bind_scores(
    scores: torch.Tensor,
    *,
    extra: torch.Tensor | None = None,
    member_ids: Iterable[int] = (),
    mode: str = "bayes",
    gain_unembed: float = 1.5,
    gain_members: float = 8.0,
    member_mass: float = 0.02,
    geom_mass: float = 0.0,
    max_strength: float = 1.0,
) -> torch.Tensor:
    """Mix the lock into next-token scores ``[..., vocab]``.

    ``bayes`` informs. ``members`` / ``unembed`` / ``both`` direct.
    """
    if mode not in ("unembed", "members", "both", "alone", "bayes"):
        raise ValueError(f"unknown prediction-bind mode {mode!r}")
    out = scores.float().clone()
    if mode == "alone":
        return out
    if mode == "bayes":
        log_pi = bind_log_prior(
            int(out.shape[-1]),
            member_ids,
            extra,
            member_mass=member_mass,
            geom_mass=geom_mass,
            device=out.device,
            dtype=out.dtype,
        )
        return bayes_inform(out, log_pi, max_strength=max_strength)
    if mode in ("unembed", "both") and extra is not None:
        add = extra.float().reshape(-1).to(device=out.device)
        if add.numel() != out.shape[-1]:
            raise ValueError(f"scores {out.shape[-1]} != unembed bind {add.numel()}")
        out = out + float(gain_unembed) * add
    if mode in ("members", "both"):
        n = out.shape[-1]
        for tid in member_ids:
            i = int(tid)
            if 0 <= i < n:
                out[..., i] = out[..., i] + float(gain_members)
    return out


def greedy_generate(
    bundle: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 48,
    extra: torch.Tensor | None = None,
    members: Sequence[str] = (),
    member_ids: Sequence[int] | None = None,
    mode: str = "alone",
    gain_unembed: float = 1.5,
    gain_members: float = 8.0,
    member_mass: float = 0.02,
    geom_mass: float = 0.0,
    max_strength: float = 1.0,
    query_only: bool = False,
    memory_vector: torch.Tensor | None = None,
    memory_gain: float = 1.0,
    live_ids: Sequence[int] | None = None,
    rival_ids: Sequence[int] | None = None,
    state_gate: bool = False,
    gate_boost: float = 6.0,
) -> str:
    """Greedy decode. ``bayes`` informs; ``query_only`` applies it at the first step.

    ``memory_vector`` inserts the lock as an extra token before the last piece.
    """
    ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
    ids = ids.to(bundle.device)
    hook = None
    if memory_vector is not None:
        from llmintent.memslot import attach_memory_slot

        ids, _pos, hook = attach_memory_slot(
            bundle, ids, memory_vector, gain=memory_gain
        )
    mids = list(member_ids) if member_ids is not None else member_token_ids(bundle, members)
    extra_dev = None
    if extra is not None:
        extra_dev = extra.detach().float().reshape(-1)

    lives = list(live_ids or [])
    rivals = list(rival_ids or [])
    gate_used = {"n": 0}

    def _step(scores: torch.Tensor) -> torch.Tensor:
        if state_gate:
            gated, fired = apply_state_gate(scores, lives, rivals, boost=gate_boost)
            if fired:
                gate_used["n"] += 1
                if gate_used["n"] >= 1:
                    return gated.to(dtype=scores.dtype)
            if gate_used["n"] >= 1:
                return scores
        bound = bind_scores(
            scores,
            extra=extra_dev,
            member_ids=mids,
            mode=mode,
            gain_unembed=gain_unembed,
            gain_members=gain_members,
            member_mass=member_mass,
            geom_mass=geom_mass,
            max_strength=max_strength,
        )
        return bound.to(dtype=scores.dtype)

    try:
        from transformers import LogitsProcessor, LogitsProcessorList

        class _Proc(LogitsProcessor):
            def __init__(self) -> None:
                super().__init__()
                self.n = 0

            def __call__(self, _input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                self.n += 1
                if query_only and self.n > 1 and not state_gate:
                    return scores
                return _step(scores)

        procs = LogitsProcessorList([_Proc()]) if (mode != "alone" or state_gate) else None
        tok = bundle.tokenizer
        eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
        end = tok.convert_tokens_to_ids("<|im_end|>") if hasattr(tok, "convert_tokens_to_ids") else None
        if isinstance(end, int) and end >= 0 and end not in eos:
            eos.append(end)
        attn = torch.ones_like(ids)
        kw: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": attn,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "logits_processor": procs,
        }
        if eos:
            kw["eos_token_id"] = eos if len(eos) > 1 else eos[0]
            if getattr(tok, "pad_token_id", None) is None:
                kw["pad_token_id"] = eos[0]
            else:
                kw["pad_token_id"] = tok.pad_token_id
        with torch.no_grad():
            out = bundle.model.generate(**kw)
        new_ids = out[0, ids.shape[1] :]
        return bundle.tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
    except Exception:
        return _greedy_loop(
            bundle,
            ids,
            max_new_tokens=max_new_tokens,
            step=_step,
            query_only=query_only,
        )
    finally:
        if hook is not None:
            hook.remove()


def _greedy_loop(
    bundle: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    step,
    query_only: bool = False,
) -> str:
    tok = bundle.tokenizer
    eos = {i for i in (tok.eos_token_id,) if i is not None}
    end = tok.convert_tokens_to_ids("<|im_end|>") if hasattr(tok, "convert_tokens_to_ids") else None
    if isinstance(end, int) and end >= 0:
        eos.add(end)
    ids = input_ids
    past = None
    new: list[int] = []
    attn = torch.ones_like(ids)
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            if past is None:
                out = bundle.model(ids, attention_mask=attn, use_cache=True)
            else:
                out = bundle.model(
                    ids[:, -1:],
                    attention_mask=torch.ones(ids.shape[0], ids.shape[1], device=ids.device),
                    past_key_values=past,
                    use_cache=True,
                )
            past = getattr(out, "past_key_values", None)
            logits = out.logits[0, -1]
            scored = logits if query_only and new else step(logits)
            tid = int(torch.argmax(scored).item())
            if tid in eos:
                break
            new.append(tid)
            nxt = torch.tensor([[tid]], device=ids.device, dtype=ids.dtype)
            ids = torch.cat([ids, nxt], dim=1)
            attn = torch.ones_like(ids)
    return tok.decode(new, skip_special_tokens=True)


def top_piece(bundle: Any, logits: torch.Tensor) -> str:
    tid = int(torch.argmax(logits.reshape(-1)).item())
    text = bundle.tokenizer.decode([tid], skip_special_tokens=True).strip()
    return text or f"id:{tid}"


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
    """Qualitative bind readout used by 27B eval scripts and tests."""
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


def mine_qual(rows: list[dict]) -> dict:
    n = len(rows) or 1
    counts: dict[str, int] = {}
    for row in rows:
        label = row.get("verdict") or "same"
        counts[label] = counts.get(label, 0) + 1
    alone_ok = sum(1 for r in rows if (r.get("alone") or {}).get("wrote_result"))
    bind_ok = sum(1 for r in rows if (r.get("members") or {}).get("wrote_result"))
    return {
        "n": len(rows),
        "counts": counts,
        "alone_ok": alone_ok / n,
        "bind_ok": bind_ok / n,
    }


def mine_bind_tracks(rows: list[dict]) -> dict:
    """Summarize operand-mass tracks from bind ablations."""

    def mean(cond: str, key: str) -> float:
        vals = [
            r[cond][key]
            for r in rows
            if cond in r and isinstance(r.get(cond), dict) and key in r[cond]
        ]
        return float(sum(vals) / len(vals)) if vals else 0.0

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


mine = mine_bind_tracks
