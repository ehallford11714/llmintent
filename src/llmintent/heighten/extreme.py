"""Extreme focused reasoning via chained forced retracements."""

from __future__ import annotations

from dataclasses import dataclass, field

from enum import Enum

from llmintent.heighten.focus import compute_focus_metrics
from llmintent.heighten.retrace import build_focused_prompt, build_retrace_prompt
from llmintent.heighten.types import FocusMetrics, RetraceMode
from llmintent.models import ModelBundle
from llmintent.trajectory import TrajectoryMapping, build_trajectory_mapping


class ExtremeRetraceMode(str, Enum):
    """Chained retrace intensities beyond standard RetraceMode."""

    DOUBLE = "extreme_double_retrace"
    TRIPLE = "extreme_triple_retrace"
    CONCEPT_LOCK = "extreme_concept_lock"
    VERIFY_CHAIN = "extreme_verify_chain"


_EXTREME_PASS_MODES = [
    RetraceMode.EXPLICIT,
    RetraceMode.CORRECTION,
    RetraceMode.CONCEPT_ANCHOR,
]

_EXTREME_SUFFIX = (
    "\n\nSTRICT FOCUS MODE: Ignore irrelevant associations. "
    "Use only {concepts}. Verify each step before proceeding. "
    "Do not commit until reasoning is concentrated on the essential chain."
)

_VERIFY_BLOCK = (
    "\nVerification pass:\n"
    "1. Restate the question in one sentence.\n"
    "2. List only essential concepts: {concepts}.\n"
    "3. Confirm the conclusion follows necessarily.\n"
    "4. Reject any step that introduces unrelated ideas."
)


@dataclass
class ExtremeRetraceChain:
    """Stored sequence of forced retracements for a prompt."""

    baseline_prompt: str
    anchor_prompt: str
    concepts: list[str]
    passes: list[str]
    combined_prompt: str
    mode: str
    retrace_layers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "baseline_prompt": self.baseline_prompt,
            "anchor_prompt": self.anchor_prompt,
            "concepts": self.concepts,
            "passes": self.passes,
            "combined_prompt": self.combined_prompt,
            "mode": self.mode,
            "retrace_layers": self.retrace_layers,
        }


def build_extreme_retrace_chain(
    anchor_prompt: str,
    *,
    baseline_prompt: str | None = None,
    concepts: list[str] | None = None,
    mode: str = ExtremeRetraceMode.TRIPLE,
    passes: int = 3,
) -> ExtremeRetraceChain:
    """
    Build chained forced retracements for extreme focused reasoning.

    Each pass appends a tighter retrace scaffold, ending in concept-lock
    or verify-chain language.
    """
    baseline = baseline_prompt or anchor_prompt
    concepts = concepts or ["the essential reasoning chain"]
    concept_str = ", ".join(concepts)

    chain: list[str] = [anchor_prompt.strip()]
    n_passes = passes
    if mode == ExtremeRetraceMode.DOUBLE:
        n_passes = 2
    elif mode in (ExtremeRetraceMode.TRIPLE, ExtremeRetraceMode.CONCEPT_LOCK, ExtremeRetraceMode.VERIFY_CHAIN):
        n_passes = max(3, passes)

    for i in range(n_passes):
        pass_mode = _EXTREME_PASS_MODES[i % len(_EXTREME_PASS_MODES)]
        segment = build_retrace_prompt(chain[-1], mode=pass_mode, concepts=concepts)
        chain.append(segment)

    combined = chain[-1]

    if mode == ExtremeRetraceMode.CONCEPT_LOCK:
        combined += _EXTREME_SUFFIX.format(concepts=concept_str)
        combined += (
            f"\nConcept lock engaged. Only process: {concept_str}. "
            "Discard all other semantic branches."
        )
    elif mode == ExtremeRetraceMode.VERIFY_CHAIN:
        combined += _VERIFY_BLOCK.format(concepts=concept_str)
        combined += _EXTREME_SUFFIX.format(concepts=concept_str)
    else:
        combined += _EXTREME_SUFFIX.format(concepts=concept_str)

    focused = build_focused_prompt(baseline, concepts=concepts)
    combined = f"{focused}\n\n{combined}"

    return ExtremeRetraceChain(
        baseline_prompt=baseline,
        anchor_prompt=anchor_prompt,
        concepts=concepts,
        passes=chain[1:],
        combined_prompt=combined,
        mode=mode,
    )


def heighten_until_focused(
    bundle: ModelBundle,
    prompt: str,
    *,
    anchor_prompt: str | None = None,
    concepts: list[str] | None = None,
    max_passes: int = 4,
    focus_threshold: float = 0.55,
    extreme: bool = False,
    transport=None,
) -> tuple[FocusMetrics, TrajectoryMapping, ExtremeRetraceChain | None, list[dict]]:
    """
    Iteratively apply forced retracements until focus_score exceeds threshold.

    Returns final focus, mapping, extreme chain (if used), and pass history.
    """
    anchor = anchor_prompt or prompt
    history: list[dict] = []
    chain: ExtremeRetraceChain | None = None
    twin = anchor if anchor != prompt else prompt

    for pass_idx in range(max_passes):
        mapping = build_trajectory_mapping(
            bundle,
            prompt,
            twin_b=twin if twin != prompt else None,
            transport=transport,
            concepts=concepts,
            include_cognitive=twin != prompt,
        )
        focus = compute_focus_metrics(mapping, focus_threshold=focus_threshold)
        history.append(
            {
                "pass": pass_idx,
                "focus_score": focus.focus_score,
                "needs_retrace": focus.needs_retrace,
                "twin_preview": twin[:120],
            }
        )
        if focus.focus_score >= focus_threshold:
            return focus, mapping, chain, history

        if extreme:
            chain = build_extreme_retrace_chain(
                anchor,
                baseline_prompt=prompt,
                concepts=concepts,
                mode=ExtremeRetraceMode.TRIPLE,
                passes=min(pass_idx + 2, 4),
            )
            twin = chain.combined_prompt
        else:
            mode = _EXTREME_PASS_MODES[pass_idx % len(_EXTREME_PASS_MODES)]
            twin = build_retrace_prompt(anchor, mode=mode, concepts=concepts or ["the core question"])

    mapping = build_trajectory_mapping(
        bundle,
        prompt,
        twin_b=twin,
        transport=transport,
        concepts=concepts,
        include_cognitive=True,
    )
    focus = compute_focus_metrics(mapping, focus_threshold=focus_threshold)
    return focus, mapping, chain, history
