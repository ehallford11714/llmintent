"""Neurotransmitter roles and testable LLM analogues.

Fly consensus_nt is a measured prediction on MaleCNS bodies. An LLM has no
acetylcholine. Analogues are computational jobs, scored separately from
biological analogy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmintent.anatomy.flycns import load_circuit_bundle, majority_nt
from llmintent.anatomy.tasks import by_split, forced_choice_logprobs


@dataclass(frozen=True)
class TransmitterAnalogue:
    fly_nt: str
    fly_role: str
    llm_operation: str
    test: str
    not_a_claim: str
    strength: str = "biological_analogy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fly_nt": self.fly_nt,
            "fly_role": self.fly_role,
            "llm_operation": self.llm_operation,
            "test": self.test,
            "not_a_claim": self.not_a_claim,
            "strength": self.strength,
        }


ANALOGUES: tuple[TransmitterAnalogue, ...] = (
    TransmitterAnalogue(
        "acetylcholine",
        "Fast excitatory write (LPLC2, T4/T5, KC, DNp01 on MaleCNS are ACh).",
        "A residual write direction whose +sign raises the matched continuation.",
        "SVD +u vs −u on looming-language action tokens; +u should win if this is the ACh analogue.",
        "No layer is acetylcholine. Token 'acetylcholine' is content selectivity, not this job.",
    ),
    TransmitterAnalogue(
        "gaba",
        "Fast inhibitory write (APL→KC is GABA and negatively signed).",
        "The opposite sign of the same axis, or a write that lowers the matched continuation.",
        "If +u is ACh-like, −u is the GABA analogue on that axis — not a second random direction.",
        "GABA is not 'the late layers'.",
    ),
    TransmitterAnalogue(
        "glutamate",
        "Inhibitory by MaleCNS/Shiu default (26/97 MBONs). Can be excitatory in Drosophila.",
        "Same subtractive class as GABA unless a glutamate-excitatory override is declared.",
        "MBON mixed-NT pool: do not treat every mushroom-body output as excitatory.",
        "Do not collapse glutamate with dopamine.",
    ),
    TransmitterAnalogue(
        "dopamine",
        "Neuromodulatory. PAM (reward) and PPL1 (aversive) are 100% dopamine onto KC.",
        "Gain on an associative mapping: scale cue→outcome without adding cue evidence.",
        "Reward-context vs aversive-context vs cue-only vs the word 'dopamine'.",
        "Mentioning dopamine is not the analogue. Fast extra excitation is the wrong model.",
    ),
    TransmitterAnalogue(
        "octopamine",
        "Neuromodulatory arousal (not in the first extracted circuits).",
        "Global gain on action logits, not a content detector.",
        "Unidentified until an octopamine-typed MaleCNS population is assayed.",
        "Not mapped onto a Qwen layer by name.",
    ),
    TransmitterAnalogue(
        "serotonin",
        "Slower contextual modulation (not in the first extracted circuits).",
        "Context axis that changes a readout without rewriting the cue.",
        "Unidentified until a serotonergic circuit is assayed.",
        "Not 'sentiment neurons'.",
    ),
)


def transmitter_report() -> dict[str, Any]:
    bundle = load_circuit_bundle()
    pops = []
    for name, row in (bundle.get("populations") or {}).items():
        nt = majority_nt(row.get("nt") or {})
        pops.append(
            {
                "population": name,
                "n": row.get("n"),
                "majority_nt": nt,
                "nt_counts": row.get("nt"),
                "llm_analogue": next(
                    (a.llm_operation for a in ANALOGUES if a.fly_nt == nt),
                    "unidentified",
                ),
            }
        )
    return {
        "source": bundle.get("retrieval"),
        "analogues": [a.to_dict() for a in ANALOGUES],
        "populations": pops,
        "edges": bundle.get("edges"),
        "note": (
            "Biological analogy, observational association, and intervention "
            "support are separate. These rows are the analogy contract."
        ),
    }


def score_value_modulation(bundle: Any | None) -> dict[str, Any]:
    """Observational test of the dopamine analogue on an LLM, if a bundle exists."""
    items = by_split("value_modulation", "confirmation")
    if bundle is None:
        return {
            "executed": False,
            "llm_source": "unavailable",
            "note": "Prepared. Not pretrained-model evidence.",
            "items": [it.to_dict() for it in items],
        }
    rows = []
    for it in items:
        row = forced_choice_logprobs(bundle, it)
        row["tags"] = list(it.tags)
        rows.append(row)

    def mean_margin(*tags: str) -> float | None:
        xs = [r["margin"] for r in rows if any(t in r.get("tags", ()) for t in tags)]
        return None if not xs else float(sum(xs) / len(xs))

    reward = mean_margin("reward_context")
    aversive = mean_margin("aversive_context")
    cue = mean_margin("cue_only")
    lexical = mean_margin("lexical_cue")
    gain_like = None
    if reward is not None and cue is not None:
        gain_like = reward - cue
    dissociated_from_word = None
    if gain_like is not None and lexical is not None:
        dissociated_from_word = abs(gain_like) > abs(lexical)
    return {
        "executed": True,
        "llm_source": "measured",
        "reward_margin": reward,
        "aversive_margin": aversive,
        "cue_only_margin": cue,
        "dopamine_word_margin": lexical,
        "reward_minus_cue": gain_like,
        "dissociated_from_word_dopamine": dissociated_from_word,
        "items": rows,
        "note": (
            "Pass if reward context shifts the cue mapping more than mentioning "
            "'dopamine', and aversive context shifts the opposite way. "
            "Observational only until an intervention gain test."
        ),
    }
