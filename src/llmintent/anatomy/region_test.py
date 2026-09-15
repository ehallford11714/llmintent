"""Analogous functional-region test in a shared operation space.

Compares fly assay selectivity to LLM candidate selectivity on matched
conditions. Never cosine-compares synapse weights to SVD coordinates.

Shared axes (both systems, when measured)
-----------------------------------------
approach     expanding loom / approaching-object language
recede       receding loom / receding language
static       static or luminance / non-approach controls
lexical_cue  content selectivity (cue word without the function)
contextual   quote / negation (computational contribution of context)
perturbation silence vs intact (causal contribution; fly simulation here)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llmintent.anatomy.fly_assay import FlyFunctionalAssay, assay_signature
from llmintent.anatomy.tasks import TaskItem, by_split, forced_choice_logprobs

SHARED_AXES = (
    "approach",
    "recede",
    "static",
    "lexical_cue",
    "contextual",
    "perturbation",
)


@dataclass
class RegionTestResult:
    function_id: str
    fly_selectivity: float
    llm_selectivity: float | None
    alignment_score: float | None
    split: str
    n_items: int
    fly_source: str
    llm_source: str
    prior_condition: str
    notes: list[str] = field(default_factory=list)
    item_rows: list[dict[str, Any]] = field(default_factory=list)
    fly_vector: dict[str, float | None] = field(default_factory=dict)
    llm_vector: dict[str, float | None] = field(default_factory=dict)
    shared_axes: list[str] = field(default_factory=list)
    content_selectivity: float | None = None
    computational_contribution: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_id": self.function_id,
            "fly_selectivity": round(self.fly_selectivity, 4),
            "llm_selectivity": None if self.llm_selectivity is None else round(self.llm_selectivity, 4),
            "alignment_score": None if self.alignment_score is None else round(self.alignment_score, 4),
            "split": self.split,
            "n_items": self.n_items,
            "fly_source": self.fly_source,
            "llm_source": self.llm_source,
            "prior_condition": self.prior_condition,
            "notes": list(self.notes),
            "items": self.item_rows,
            "fly_vector": self.fly_vector,
            "llm_vector": self.llm_vector,
            "shared_axes": list(self.shared_axes),
            "content_selectivity": self.content_selectivity,
            "computational_contribution": self.computational_contribution,
        }


def _mean(xs: list[float]) -> float | None:
    return None if not xs else float(np.mean(xs))


def fly_condition_vector(assay: FlyFunctionalAssay) -> dict[str, float | None]:
    """Map assay readouts onto the shared operation axes."""
    intact = [r for r in assay.results if r.perturbation == "none"]
    approach = [r.readout for r in intact if r.stimulus.startswith("expanding")]
    recede = [r.readout for r in intact if r.stimulus.startswith("receding")]
    static = [
        r.readout
        for r in intact
        if r.stimulus.startswith("static") or r.stimulus.startswith("luminance")
    ]
    silenced = [r.readout for r in assay.results if r.perturbation == "silence_LPLC2"]
    intact_match = [
        r.readout
        for r in assay.results
        if r.perturbation == "none" and r.held_out and r.stimulus.startswith("expanding")
    ]
    perturb = None
    if intact_match and silenced:
        perturb = float(np.mean(intact_match) - np.mean(silenced))
    return {
        "approach": _mean(approach),
        "recede": _mean(recede),
        "static": _mean(static),
        "lexical_cue": None,  # fly circuit has no lexical cue axis
        "contextual": None,
        "perturbation": perturb,
    }


def llm_condition_vector(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Map forced-choice margins onto the same axes via item tags."""

    def tagged(*need: str) -> list[float]:
        out = []
        for r in rows:
            tags = tuple(r.get("tags") or ())
            if any(t in tags for t in need):
                out.append(float(r["margin"]))
        return out

    return {
        "approach": _mean(tagged("positive")),
        "recede": _mean(tagged("receding", "negative")),
        "static": _mean(tagged("auditory_control", "benign_shortcut")),
        "lexical_cue": _mean(tagged("lexical_cue", "content_selectivity")),
        "contextual": _mean(tagged("quote", "negation", "negation_like")),
        "perturbation": None,  # requires an intervention run, not a clean score
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def align_vectors(
    fly_vec: dict[str, float | None],
    llm_vec: dict[str, float | None],
) -> tuple[float | None, list[str]]:
    shared = [
        ax
        for ax in SHARED_AXES
        if fly_vec.get(ax) is not None and llm_vec.get(ax) is not None
    ]
    pearson = _pearson(
        [float(fly_vec[ax]) for ax in shared],
        [float(llm_vec[ax]) for ax in shared],
    )
    signed = None
    if fly_vec.get("approach") is not None and fly_vec.get("recede") is not None:
        if llm_vec.get("approach") is not None and llm_vec.get("recede") is not None:
            fly_sel = float(fly_vec["approach"]) - float(fly_vec["recede"])
            llm_sel = float(llm_vec["approach"]) - float(llm_vec["recede"])
            signed = float(np.tanh(llm_sel) * np.sign(fly_sel or 1.0))
    # Pearson needs ≥3 axes; otherwise use same-sign approach−recede selectivity.
    if len(shared) >= 3 and pearson is not None:
        return pearson, shared
    return signed, shared


def _selectivity(rows: list[dict[str, Any]], positive_tag: str, negative_tag: str) -> float:
    pos = [r["margin"] for r in rows if positive_tag in r.get("tags", ())]
    neg = [r["margin"] for r in rows if negative_tag in r.get("tags", ())]
    if not pos:
        pos = [r["margin"] for r in rows if r.get("correct")]
    if not neg:
        neg = [r["margin"] for r in rows if not r.get("correct")]
    if not pos or not neg:
        return float(np.mean([r["margin"] for r in rows])) if rows else 0.0
    return float(np.mean(pos) - np.mean(neg))


def _content_vs_computation(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    cue = [r["margin"] for r in rows if "lexical_cue" in r.get("tags", ()) or "content_selectivity" in r.get("tags", ())]
    functional = [
        r["margin"]
        for r in rows
        if "positive" in r.get("tags", ()) and "lexical_cue" not in r.get("tags", ())
    ]
    contextual = [
        r["margin"]
        for r in rows
        if any(t in r.get("tags", ()) for t in ("quote", "negation", "negation_like"))
    ]
    content = _mean(cue)
    computation = None
    if functional and contextual:
        computation = float(np.mean(functional) - np.mean(contextual))
    elif functional:
        computation = float(np.mean(functional))
    return content, computation


def test_analogous_region(
    assay: FlyFunctionalAssay,
    bundle: Any | None,
    items: list[TaskItem],
    *,
    prior_condition: str = "fly_informed",
) -> RegionTestResult:
    """Held-out functional comparison in the shared approaching-object → action space."""
    sig = assay_signature(assay)
    fly_vec = fly_condition_vector(assay)
    rows: list[dict[str, Any]] = []
    llm_source = "unavailable"
    if bundle is not None:
        llm_source = "measured"
        for it in items:
            row = forced_choice_logprobs(bundle, it)
            row["tags"] = list(it.tags)
            rows.append(row)
    llm_vec = llm_condition_vector(rows) if rows else {ax: None for ax in SHARED_AXES}
    llm_sel = _selectivity(rows, "positive", "negative") if rows else None
    fly_sel = float(sig["selectivity"])
    align, shared = align_vectors(fly_vec, llm_vec) if rows else (None, [])
    content, computation = _content_vs_computation(rows)
    notes = [
        sig["shared_operation"],
        "No cosine between fly synapse weights and LLM tensors.",
        f"Prior condition={prior_condition}",
        "Alignment is Pearson correlation over shared functional axes, not tensor cosine.",
        "Content selectivity (cue-word items) is scored separately from computational contribution.",
    ]
    if bundle is None:
        notes.append("LLM bundle unavailable; region test prepared, not executed on a checkpoint.")
    if align is None and rows:
        notes.append("Alignment unidentified: fewer than two shared measured axes.")
    return RegionTestResult(
        function_id=assay.function_id,
        fly_selectivity=fly_sel,
        llm_selectivity=llm_sel,
        alignment_score=align,
        split="confirmation" if items and items[0].split == "confirmation" else (items[0].split if items else "none"),
        n_items=len(items),
        fly_source=assay.evidence_mode,
        llm_source=llm_source,
        prior_condition=prior_condition,
        notes=notes,
        item_rows=rows,
        fly_vector=fly_vec,
        llm_vector=llm_vec,
        shared_axes=shared,
        content_selectivity=content,
        computational_contribution=computation,
    )


def compare_priors(
    assay: FlyFunctionalAssay,
    bundle: Any | None,
    items: list[TaskItem],
    *,
    fly_top_contrast: float | None = None,
    no_prior_top_contrast: float | None = None,
    shuffled_top_contrast: float | None = None,
    rewired_top_contrast: float | None = None,
) -> dict[str, Any]:
    """Comparable-budget fly-informed vs no-prior vs shuffled vs rewired.

    The unmodified-model region test is the same items either way. The prior is
    evaluated on which SVD candidate it nominates (held-out contrast), not on
    painting depth bands as discovered organization.
    """
    behavior = test_analogous_region(assay, bundle, items, prior_condition="shared_confirmation")
    scores = {
        "fly_informed": fly_top_contrast,
        "no_prior": no_prior_top_contrast,
        "shuffled_labels": shuffled_top_contrast,
        "rewired_graph": rewired_top_contrast,
    }
    executed = {k: v for k, v in scores.items() if v is not None}
    improved = None
    if "fly_informed" in executed and any(k != "fly_informed" for k in executed):
        others = [v for k, v in executed.items() if k != "fly_informed"]
        improved = executed["fly_informed"] > max(others)
    return {
        "scores": scores,
        "fly_prior_improved": improved,
        "unmodified_confirmation": behavior.to_dict(),
        "note": (
            "Priors compared on SVD candidate contrast with the same top_k. "
            "Unmodified confirmation accuracy is reported separately and is not "
            "the prior comparison."
        ),
    }


def synthetic_llm_rows(items: list[TaskItem], *, mode: str, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic fixture scorer. Never counts as pretrained-model evidence."""
    rng = np.random.default_rng(seed)
    base: list[dict[str, Any]] = []
    for it in items:
        tags = set(it.tags)
        if "positive" in tags:
            margin = 2.0
        elif "receding" in tags or "negative" in tags:
            margin = -1.0
        elif "lexical_cue" in tags or "content_selectivity" in tags:
            margin = 0.15
        elif "quote" in tags or "negation" in tags or "negation_like" in tags:
            margin = -0.4
        else:
            margin = 0.0
        base.append(
            {
                "item_id": it.id,
                "family": it.family,
                "split": it.split,
                "margin": margin,
                "correct": margin > 0,
                "tags": list(it.tags),
                "source": "synthetic",
                "scorer": f"region_test_fixture/{mode}",
            }
        )
    if mode == "matched":
        return base
    if mode == "anti":
        out = []
        for row in base:
            flipped = dict(row)
            flipped["margin"] = -float(row["margin"])
            flipped["correct"] = flipped["margin"] > 0
            out.append(flipped)
        return out
    if mode == "shuffled":
        margins = [float(r["margin"]) for r in base]
        rng.shuffle(margins)
        out = []
        for row, m in zip(base, margins):
            shuffled = dict(row)
            shuffled["margin"] = m
            shuffled["correct"] = m > 0
            shuffled["scorer"] = "region_test_fixture/shuffled"
            out.append(shuffled)
        return out
    raise ValueError(mode)


def score_synthetic_alignment(
    assay: FlyFunctionalAssay,
    items: list[TaskItem],
    *,
    mode: str,
    seed: int = 0,
) -> dict[str, Any]:
    fly_vec = fly_condition_vector(assay)
    rows = synthetic_llm_rows(items, mode=mode, seed=seed)
    llm_vec = llm_condition_vector(rows)
    align, shared = align_vectors(fly_vec, llm_vec)
    return {
        "mode": mode,
        "alignment": align,
        "shared_axes": shared,
        "fly_vector": fly_vec,
        "llm_vector": llm_vec,
        "source": "synthetic",
        "n_items": len(items),
    }


def validate_analogous_region_test(
    *,
    assay: FlyFunctionalAssay | None = None,
    split: str = "confirmation",
    seed: int = 0,
) -> dict[str, Any]:
    """Mechanics check of the analogous-region test. Not pretrained-model evidence.

    A matched synthetic scorer that prefers approaching-object items must align
    with the fly approach>recede signature more than a shuffled or sign-flipped
    scorer. If that fails, the shared-space mapping itself is broken.
    """
    from llmintent.anatomy.fly_assay import run_looming_giant_fibre_assay

    assay = assay or run_looming_giant_fibre_assay()
    items = by_split("looming_language", split)  # type: ignore[arg-type]
    matched = score_synthetic_alignment(assay, items, mode="matched", seed=seed)
    shuffled = score_synthetic_alignment(assay, items, mode="shuffled", seed=seed)
    anti = score_synthetic_alignment(assay, items, mode="anti", seed=seed)
    m = matched["alignment"]
    s = shuffled["alignment"]
    a = anti["alignment"]
    mechanics_pass = (
        m is not None
        and (s is None or m > s)
        and (a is None or m > a)
        and m > 0
    )
    prepared = test_analogous_region(assay, None, items, prior_condition="validation_no_checkpoint")
    return {
        "executed": True,
        "source": "synthetic",
        "note": (
            "Fixture scorers validate the shared-space comparison. "
            "They are not evidence about any pretrained model."
        ),
        "matched": matched,
        "shuffled": shuffled,
        "anti": anti,
        "mechanics_pass": bool(mechanics_pass),
        "no_checkpoint_region_test": prepared.to_dict(),
        "fly_assay": {
            "circuit_id": assay.circuit_id,
            "evidence_mode": assay.evidence_mode,
            "living_fly": assay.living_fly,
            "validation_status": assay.validation_status,
        },
    }
