"""Extensible computational functions. Fly atlas eleven is one hypothesis set."""

from __future__ import annotations

from dataclasses import dataclass

from llmintent.anatomy.atlas import REGIONS

FUNCTION_IDS = (
    "feature_discrimination",
    "integration",
    "associative_retrieval",
    "contextual_modulation",
    "evidence_accumulation",
    "intention_of_actuation",
    "reflection_of_actuation",
    "withhold_actuation",
    "action_selection",
    "output_control",
    "visual_language_processing",
    "value_modulation",
    "signed_write",
)


@dataclass(frozen=True)
class ComputationalFunction:
    id: str
    description: str
    observable: str
    prediction: str
    fly_prior: tuple[str, ...] = ()
    llm_analogue: str | None = None
    modality_note: str = ""
    grouping: str = "computational_v1"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "observable": self.observable,
            "prediction": self.prediction,
            "fly_prior": list(self.fly_prior),
            "llm_analogue": self.llm_analogue,
            "modality_note": self.modality_note,
            "grouping": self.grouping,
        }


FUNCTIONS: tuple[ComputationalFunction, ...] = (
    ComputationalFunction(
        id="feature_discrimination",
        description="Select one sensory or lexical feature family from others.",
        observable="Higher activity / logprob for matched features than for length-matched distractors.",
        prediction="A nominated component responds more to positive feature items than to distractors.",
        fly_prior=("vision", "auditory", "olfactory", "gustatory", "somatosensory"),
        llm_analogue="candidate: early residual / embedding subspaces",
        modality_note=(
            "On a text-only model, scene descriptions are visual-language processing, "
            "not visual perception."
        ),
    ),
    ComputationalFunction(
        id="visual_language_processing",
        description="Track described scenes, motion, and spatial language in text.",
        observable="Selectivity to visual-language items without claiming photoreception.",
        prediction="Components recruited by looming/dark/shape language fail on matched auditory language.",
        fly_prior=("vision",),
        llm_analogue="candidate association only until intervention",
        modality_note="Text-only: do not label this visual perception.",
    ),
    ComputationalFunction(
        id="integration",
        description="Combine two cue families into one buffer.",
        observable="Conjunction items recruit a component that neither cue family recruits alone.",
        prediction="Ablating the component hurts two-cue items more than single-cue items.",
        fly_prior=("workspace",),
    ),
    ComputationalFunction(
        id="associative_retrieval",
        description="Retrieve a bound outcome from a cue, not merely mention memory words.",
        observable="Correct retrieved continuation without the cue word remember.",
        prediction="Mentioning 'remember' is not sufficient; omitted-cue retrieval still uses the region.",
        fly_prior=("associative",),
    ),
    ComputationalFunction(
        id="contextual_modulation",
        description="Change a readout depending on a context token (negation, quote, language).",
        observable="Sign flip or suppression under negation / quotation relative to matched affirmatives.",
        prediction="The same cue word does not recruit the function when quoted or negated.",
        fly_prior=("valence", "causal_logic"),
    ),
    ComputationalFunction(
        id="evidence_accumulation",
        description="Use a cause to pick a consequence. Distinct from detecting the word because.",
        observable="Forced-choice accuracy on cue-free causal items above chance.",
        prediction="A region that only fires on 'because' will fail cue-free causal items.",
        fly_prior=("causal_logic",),
    ),
    ComputationalFunction(
        id="intention_of_actuation",
        description="Commit to an action that has not yet been taken.",
        observable="Residual closer to an intend-to-act probe than to an already-acted probe.",
        prediction="Approaching-threat language with a future-act frame raises intention, not reflection.",
        fly_prior=("descending",),
        llm_analogue="candidate: late residual / action-token subspace",
        modality_note="Not a fly escape. Intention of actuation is a language-model operation.",
    ),
    ComputationalFunction(
        id="reflection_of_actuation",
        description="Represent an action that already occurred or is being reviewed.",
        observable="Residual closer to an already-acted probe than to an intend-to-act probe.",
        prediction="After-the-fact jump/stop language raises reflection, not intention.",
        fly_prior=("descending", "workspace"),
        modality_note="Reflection of actuation is not motor output and not a fly function.",
    ),
    ComputationalFunction(
        id="withhold_actuation",
        description="Choose not to act; wait or watch instead of committing.",
        observable="Residual closer to withhold than to intention on recede/wait frames.",
        prediction="Receding or wait language should not score as intention of actuation.",
        fly_prior=("descending",),
    ),
    ComputationalFunction(
        id="action_selection",
        description="Commit to one action token family (stop vs go) given a constraint.",
        observable="Forced-choice prefers the instructed action on held-out paraphrases.",
        prediction="Mentioning 'stop' in a harmless quote does not by itself select the action.",
        fly_prior=("descending",),
    ),
    ComputationalFunction(
        id="output_control",
        description="Format and emit the chosen continuation.",
        observable="Late-layer / unembed alignment with the selected token.",
        prediction="Intervening only at the unembed changes surface form more than earlier computation.",
        fly_prior=("motor",),
    ),
    ComputationalFunction(
        id="value_modulation",
        description=(
            "Scale an associative mapping by a value context (reward vs aversive), "
            "without being the cue detector itself. Fly analogue: PAM vs PPL1 dopamine onto KC."
        ),
        observable=(
            "Cue+reward context changes the continuation margin more than cue-only, "
            "and more than a sentence that merely mentions dopamine."
        ),
        prediction=(
            "A dopamine analogue is a gain on retrieval, not an extra excitatory write "
            "and not the word 'dopamine'."
        ),
        fly_prior=("associative",),
        llm_analogue="multiplicative residual gain on an associative subspace",
        modality_note="Dopamine is neuromodulatory in the fly; do not treat it as fast acetylcholine.",
    ),
    ComputationalFunction(
        id="signed_write",
        description="Add to vs subtract from a residual (ACh/glutamate-exc vs GABA).",
        observable="Opposite SVD signs move action logits in opposite directions.",
        prediction="The minus sign of an ACh-like write behaves like a GABA analogue on the same axis.",
        fly_prior=("vision", "associative"),
        llm_analogue="signed residual write; not a named neurotransmitter in the weights",
    ),
)

FUNCTION_BY_ID = {f.id: f for f in FUNCTIONS}


def fly_atlas_hypothesis_set() -> list[dict]:
    """Original eleven regions, retained as one labeled grouping."""
    return [
        {
            "id": r.id,
            "grouping": "fly_atlas_v1",
            "handles": r.handles,
            "fly_neuropil": r.fly_neuropil,
            "band": r.band,
            "depth_prior": list(r.depth),
            "depth_prior_note": "Optional depth prior/baseline, not a discovered organization.",
        }
        for r in REGIONS
    ]


def functions_to_dict() -> dict:
    return {
        "grouping": "computational_v1",
        "functions": [f.to_dict() for f in FUNCTIONS],
        "alternate_groupings": ["fly_atlas_v1"],
        "unidentified_llm_analogue_allowed": True,
        "fly_atlas_v1": fly_atlas_hypothesis_set(),
    }
