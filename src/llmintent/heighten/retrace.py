"""Retrace prompt scaffolding — force the model to reconsider its reasoning path."""

from __future__ import annotations

from llmintent.heighten.types import RetraceMode, RetracePlan
from llmintent.trajectory import TrajectoryMapping


_RETRACE_TEMPLATES = {
    RetraceMode.EXPLICIT: (
        "{anchor}\n"
        "Wait — let me retrace my reasoning step by step, focusing only on what matters.\n"
        "Step 1: Restate the problem clearly.\n"
        "Step 2: Identify the key concepts: {concepts}.\n"
        "Step 3: Re-derive the answer carefully."
    ),
    RetraceMode.CONCEPT_ANCHOR: (
        "{anchor}\n"
        "Focusing strictly on {concepts}, let me work through this again:\n"
        "First, what is being asked? Second, which concepts apply? Third, the conclusion."
    ),
    RetraceMode.PIVOT_REPLAY: (
        "{anchor}\n"
        "Returning to the core question before committing to an answer — "
        "replaying from the inference point with focus on {concepts}."
    ),
    RetraceMode.CORRECTION: (
        "{anchor}\n"
        "I need to reconsider. My prior path may have been diffuse. "
        "Let me correct course and reason again, step by step, about {concepts}."
    ),
    RetraceMode.FOCUSED_COT: (
        "{anchor}\n"
        "Let's think step by step, but only along the essential reasoning chain.\n"
        "Key concepts: {concepts}.\n"
        "Step 1 — parse. Step 2 — compute. Step 3 — verify."
    ),
}

_FOCUSED_TEMPLATE = (
    "{prompt}\n"
    "Answer with focused reasoning on {concepts} only. "
    "Do not speculate; commit only after verifying each step."
)


def build_retrace_prompt(
    anchor_prompt: str,
    *,
    mode: RetraceMode = RetraceMode.EXPLICIT,
    concepts: list[str] | None = None,
) -> str:
    concepts = concepts or ["the core question"]
    concept_str = ", ".join(concepts)
    template = _RETRACE_TEMPLATES[mode]
    return template.format(anchor=anchor_prompt.strip(), concepts=concept_str)


def build_focused_prompt(
    prompt: str,
    *,
    concepts: list[str] | None = None,
) -> str:
    concepts = concepts or ["the essential reasoning steps"]
    return _FOCUSED_TEMPLATE.format(prompt=prompt.strip(), concepts=", ".join(concepts))


def plan_retrace(
    baseline_prompt: str,
    anchor_prompt: str,
    mapping: TrajectoryMapping,
    *,
    mode: RetraceMode = RetraceMode.EXPLICIT,
    concepts: list[str] | None = None,
) -> RetracePlan:
    """
    Build a retrace plan from baseline trajectory pivots and concepts.

    Retrace layers = workspace band + reasoning peaks + inference pivot.
    """
    concepts = concepts or []
    if not concepts:
        concepts = _infer_concepts_from_mapping(mapping)

    retrace_layers = sorted(set(mapping.pivots.values()))
    if "reasoning" in mapping.layers.columns:
        top = mapping.layers.nlargest(3, "reasoning")["layer"].astype(int).tolist()
        retrace_layers = sorted(set(retrace_layers + top))

    retrace_prompt = build_retrace_prompt(anchor_prompt, mode=mode, concepts=concepts)
    focused_prompt = build_focused_prompt(baseline_prompt, concepts=concepts)

    rationale = (
        f"Mode={mode.value}; pivots={mapping.pivots}; "
        f"retrace targets layers {retrace_layers}; concepts={concepts}"
    )

    return RetracePlan(
        mode=mode,
        baseline_prompt=baseline_prompt,
        anchor_prompt=anchor_prompt,
        retrace_prompt=retrace_prompt,
        focused_prompt=focused_prompt,
        concepts=concepts,
        retrace_layers=retrace_layers,
        rationale=rationale,
    )


def _infer_concepts_from_mapping(mapping: TrajectoryMapping) -> list[str]:
    concepts = list(mapping.concept_hits.keys())
    if concepts:
        return concepts[:5]
    if "top_intent" in mapping.layers.columns:
        intents = mapping.layers["top_intent"].astype(str).tolist()
        unique = [i for i in dict.fromkeys(intents) if i and i.strip()][:5]
        if unique:
            return unique
    return ["the problem", "the reasoning steps"]
