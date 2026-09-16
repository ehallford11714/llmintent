"""Independently scored task families tied to tested fly operations.

Content selectivity ≠ computational contribution.
Synthetic fixture labels never count as evidence about a pretrained model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from llmintent.anatomy.evidence import hash_prompt

Split = Literal["discovery", "calibration", "confirmation"]


@dataclass(frozen=True)
class TaskItem:
    id: str
    family: str
    split: Split
    prompt: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    tags: tuple[str, ...]
    template_group: str
    cue_word: bool
    language: str = "en"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "split": self.split,
            "prompt": self.prompt,
            "positive": list(self.positive),
            "negative": list(self.negative),
            "tags": list(self.tags),
            "template_group": self.template_group,
            "cue_word": self.cue_word,
            "language": self.language,
            "prompt_hash": hash_prompt(self.prompt),
        }


def _item(fid, family, split, prompt, pos, neg, tags, group, cue, lang="en") -> TaskItem:
    return TaskItem(fid, family, split, prompt, tuple(pos), tuple(neg), tuple(tags), group, cue, lang)


def looming_language_family() -> tuple[TaskItem, ...]:
    """Tied to the fly LPLC2/DNp01 assay: approaching-object evidence → action."""
    pos, neg = (" stop", " jump", " escape"), (" sing", " wait", " smile")
    return (
        _item("loom_d0", "looming_language", "discovery",
              "A dark shape is rushing toward the viewer. The next action is to",
              pos, neg, ("positive", "visual_language"), "approach_en", False),
        _item("loom_d1", "looming_language", "discovery",
              "An expanding shadow fills the field. One should",
              pos, neg, ("positive", "visual_language"), "approach_en", False),
        _item("loom_d2", "looming_language", "discovery",
              "The word looming appears in this sentence. The next action is to",
              pos, neg, ("lexical_cue", "content_selectivity"), "cue_en", True),
        _item("loom_d3", "looming_language", "discovery",
              "I hear a song in the air. The next action is to",
              (" listen", " smile"), pos, ("negative", "auditory_control"), "audio_en", False),
        _item("loom_c0", "looming_language", "calibration",
              "A large object is approaching quickly. The next action is to",
              pos, neg, ("positive", "paraphrase"), "approach_en", False),
        _item("loom_k0", "looming_language", "confirmation",
              "Something big is coming straight at me. The next action is to",
              pos, neg, ("positive", "held_out"), "approach_en_holdout", False),
        _item("loom_k1", "looming_language", "confirmation",
              "The object is receding into the distance. The next action is to",
              (" wait", " watch"), pos, ("negative", "receding"), "recede_en_holdout", False),
        _item("loom_k2", "looming_language", "confirmation",
              "He said \"a dark shape is rushing toward you\" as a joke. The next action is to",
              (" laugh", " wait"), pos, ("quote", "negation_like"), "quote_en_holdout", False),
        _item("loom_k3", "looming_language", "confirmation",
              "No dark shape is rushing toward anyone. The next action is to",
              (" wait", " stay"), pos, ("negation",), "neg_en_holdout", False),
        _item("loom_k4", "looming_language", "confirmation",
              "Una sombra oscura se abalanza. La siguiente accion es",
              (" parar", " saltar", " stop"), (" cantar", " sonreir"), ("positive", "multilingual", "es"), "approach_es_holdout", False, "es"),
        _item("loom_k5", "looming_language", "confirmation",
              "I see a car. Stop.",
              (" stop",), (" accelerate", " sing"), ("benign_shortcut",), "car_en_holdout", False),
        _item("loom_k6", "looming_language", "confirmation",
              "I see a car because the light is red. Stop.",
              (" stop",), (" accelerate",), ("benign_because",), "car_because_en_holdout", True),
    )


def causal_family() -> tuple[TaskItem, ...]:
    pos, neg = (" boils", " melts"), (" freezes", " sings")
    return (
        _item("cau_d0", "causal_consequence", "discovery",
              "If water is heated to 100C it", pos, neg, ("positive",), "heat_en", True),
        _item("cau_d1", "causal_consequence", "discovery",
              "Water was heated to 100 degrees. Next it", pos, neg, ("positive", "cue_free"), "heat_en", False),
        _item("cau_d2", "causal_consequence", "discovery",
              "The sentence contains because. Water at 100C", pos, neg, ("lexical_cue",), "cue_en", True),
        _item("cau_k0", "causal_consequence", "confirmation",
              "Ice left in a hot pan", (" melts", " boils"), (" freezes",), ("positive",), "ice_en_holdout", False),
        _item("cau_k1", "causal_consequence", "confirmation",
              "He wrote \"heated to 100C because science\". Water then",
              (" boils",), (" freezes",), ("quote",), "quote_en_holdout", True),
        _item("cau_k2", "causal_consequence", "confirmation",
              "El agua se calienta a 100C. Entonces",
              (" hierve", " boils"), (" congela",), ("multilingual", "es"), "heat_es_holdout", False, "es"),
    )


def value_modulation_family() -> tuple[TaskItem, ...]:
    """Dopamine analogue: value context scales cue→action, vs the word dopamine."""
    take, avoid = (" take", " approach", " accept"), (" avoid", " reject", " flee")
    return (
        _item("val_d0", "value_modulation", "discovery",
              "The cue previously predicted a reward. Given the cue, one should",
              take, avoid, ("reward_context", "positive"), "reward_en", False),
        _item("val_d1", "value_modulation", "discovery",
              "The cue previously predicted a punishment. Given the cue, one should",
              avoid, take, ("aversive_context",), "aversive_en", False),
        _item("val_d2", "value_modulation", "discovery",
              "The sentence contains the word dopamine. Next one should",
              take, avoid, ("lexical_cue", "content_selectivity"), "word_en", True),
        _item("val_k0", "value_modulation", "confirmation",
              "That signal used to mean something good. Seeing it again, one should",
              take, avoid, ("reward_context", "held_out"), "reward_en_holdout", False),
        _item("val_k1", "value_modulation", "confirmation",
              "That signal used to mean something bad. Seeing it again, one should",
              avoid, take, ("aversive_context", "held_out"), "aversive_en_holdout", False),
        _item("val_k2", "value_modulation", "confirmation",
              "The same cue is present with no value history. One should",
              (" wait", " look"), take, ("cue_only",), "cue_en_holdout", False),
        _item("val_k3", "value_modulation", "confirmation",
              "He said \"dopamine reward\" as a joke. Next one should",
              (" laugh", " wait"), take, ("quote", "lexical_cue"), "quote_en_holdout", True),
    )


def all_items() -> tuple[TaskItem, ...]:
    return looming_language_family() + causal_family() + value_modulation_family()


def by_split(family: str, split: Split) -> list[TaskItem]:
    return [it for it in all_items() if it.family == family and it.split == split]


def holdout_items(family: str, *, discovery_split: Split = "discovery") -> list[TaskItem]:
    """Confirmation items whose template group never appeared in discovery."""
    discovery_groups = {it.template_group for it in by_split(family, discovery_split)}
    return [
        it
        for it in by_split(family, "confirmation")
        if it.template_group not in discovery_groups
    ]


LOOM_ACTION = (" stop", " jump", " escape", " parar", " saltar")
LOOM_NONACTION = (
    " wait", " watch", " smile", " sing", " listen", " stay", " laugh",
    " cantar", " sonreir",
)
VALUE_ACTION = (" take", " approach", " accept")
VALUE_NONACTION = (" avoid", " reject", " flee")


def family_action_tokens(family: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Escape/approach-oriented tokens, independent of which answer is correct."""
    if family == "looming_language":
        return LOOM_ACTION, LOOM_NONACTION
    if family == "value_modulation":
        return VALUE_ACTION, VALUE_NONACTION
    return (), ()


def _norm_cont(text: str) -> str:
    return (text or "").strip().lower()


def oriented_action_score(item: TaskItem, scored: dict[str, float]) -> float:
    """Best action-token logprob minus best non-action logprob.

    Receding items keep wait as the *correct* answer (margin), but wait is
    still a non-action. Alignment must not treat a correct wait as escape.
    """
    action, nonaction = family_action_tokens(item.family)
    if not action:
        if not scored:
            return 0.0
        return float(max(scored.values()) - min(scored.values()))
    act_key = {_norm_cont(t) for t in action}
    non_key = {_norm_cont(t) for t in nonaction}
    act_vals = [v for k, v in scored.items() if _norm_cont(k) in act_key]
    non_vals = [v for k, v in scored.items() if _norm_cont(k) in non_key]
    if not act_vals or not non_vals:
        pos = item.positive[0] if item.positive else ""
        return float(scored.get(pos, 0.0))
    return float(max(act_vals) - max(non_vals))


def forced_choice_logprobs(bundle: Any, item: TaskItem) -> dict[str, Any]:
    """Deterministic scorer: mean logprob of positive vs negative continuations."""
    import torch
    from llmintent.forward import encode_prompt_ids

    tok = bundle.tokenizer
    ids = encode_prompt_ids(bundle, item.prompt, thinking=False)

    def _score(cont: str) -> float:
        extra = tok(cont, add_special_tokens=False, return_tensors="pt")["input_ids"].to(bundle.device)
        full = torch.cat([ids, extra], dim=1)
        with torch.no_grad():
            out = bundle.model(full)
            logits = out.logits[0, ids.shape[1] - 1 : full.shape[1] - 1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            chosen = extra[0]
            n = min(int(chosen.shape[0]), int(logp.shape[0]))
            if n <= 0:
                return -1e9
            return float(logp[torch.arange(n, device=logp.device), chosen[:n]].mean().cpu())

    pos = [_score(c) for c in item.positive]
    neg = [_score(c) for c in item.negative]
    best_pos, best_neg = max(pos), max(neg)
    margin = best_pos - best_neg
    scored = {c: s for c, s in zip(item.positive, pos)}
    scored.update({c: s for c, s in zip(item.negative, neg)})
    return {
        "item_id": item.id,
        "family": item.family,
        "split": item.split,
        "positive_logprob": best_pos,
        "negative_logprob": best_neg,
        "margin": margin,
        "action_score": oriented_action_score(item, scored),
        "accuracy": 1.0 if margin > 0 else 0.0,
        "correct": best_pos > best_neg,
        "source": "measured",
        "scorer": "forced_choice_mean_logprob/v1",
        "cue_word": item.cue_word,
        "language": item.language,
        "note": "margin is correctness; action_score is escape/approach oriented.",
    }


def family_baseline(bundle: Any, family: str, split: Split = "discovery") -> dict[str, Any]:
    items = by_split(family, split)
    rows = [forced_choice_logprobs(bundle, it) for it in items]
    acc = sum(1 for r in rows if r["correct"]) / max(len(rows), 1)
    return {
        "family": family,
        "split": split,
        "n": len(rows),
        "accuracy": acc,
        "mean_action_score": sum(r.get("action_score", 0.0) for r in rows) / max(len(rows), 1),
        "chance": 0.5,
        "above_chance": acc > 0.5,
        "items": rows,
        "note": "If not above chance, do not interpret a failed probe as absence of the function.",
    }
