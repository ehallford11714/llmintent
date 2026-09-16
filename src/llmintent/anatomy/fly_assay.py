"""Fly functional assay contract and the first looming → giant-fibre circuit.

Evidence modes
--------------
* ``recording_reanalysis`` — published recordings re-scored here.
* ``published_simulation_reproduction`` — replay of a published simulator.
* ``proposed_simulation`` — a new connectome-constrained dynamics model.

A simulated ablation is never reported as an experiment on a living fly.
Observational recordings do not by themselves establish knockout effects.

This workspace's ``fly-brain`` package provides a literature-core type graph
and a pixel looming *proxy* (``flybrain.vision.statistics``), not the
Lappalainen et al. 2024 connectome-constrained optic-lobe network and not a
MaleCNS synapse dump. The assay below is a proposed rate-model that aims to
reproduce the published *direction* of the LPLC2 → DNp01 looming-escape
result (von Reyn et al. 2014; Klapoetke et al. 2017) on held-out loom speeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from llmintent.anatomy.evidence import SCHEMA_VERSION, utc_now

EvidenceMode = Literal[
    "recording_reanalysis",
    "published_simulation_reproduction",
    "proposed_simulation",
]


@dataclass
class FlyStimulus:
    name: str
    kind: str
    params: dict[str, float]
    held_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "params": dict(self.params),
            "held_out": self.held_out,
        }


@dataclass
class FlyPerturbation:
    target: str
    kind: str  # silence | rescue | none | random_control
    scale: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "kind": self.kind, "scale": self.scale}


@dataclass
class FlyAssayResult:
    stimulus: str
    held_out: bool
    rates: dict[str, float]
    readout: float
    perturbation: str
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimulus": self.stimulus,
            "held_out": self.held_out,
            "rates": {k: round(v, 4) for k, v in self.rates.items()},
            "readout": round(self.readout, 4),
            "perturbation": self.perturbation,
            "score": round(self.score, 4),
            "source": self.source,
        }


@dataclass
class FlyFunctionalAssay:
    """One specified fly circuit experiment or simulation."""

    circuit_id: str
    specimen: str
    dataset_version: str
    evidence_mode: EvidenceMode
    function_id: str
    neurons: tuple[str, ...]
    citations: tuple[str, ...]
    synaptic_sign_assumptions: dict[str, str]
    delays_ms: dict[str, float]
    stimulus_encoding: str
    readout: str
    validation_status: str
    notes: list[str] = field(default_factory=list)
    results: list[FlyAssayResult] = field(default_factory=list)
    contributors: list[str] = field(default_factory=list)
    fit_conditions: str | None = None
    eval_conditions: str | None = None
    living_fly: bool = False
    created_at: str = field(default_factory=utc_now)
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "circuit_id": self.circuit_id,
            "specimen": self.specimen,
            "dataset_version": self.dataset_version,
            "evidence_mode": self.evidence_mode,
            "function_id": self.function_id,
            "neurons": list(self.neurons),
            "citations": list(self.citations),
            "synaptic_sign_assumptions": dict(self.synaptic_sign_assumptions),
            "delays_ms": dict(self.delays_ms),
            "stimulus_encoding": self.stimulus_encoding,
            "readout": self.readout,
            "validation_status": self.validation_status,
            "simulator_mechanics": self.validation_status,
            "biological_evidence": (
                "established"
                if self.living_fly or self.evidence_mode == "recording_reanalysis"
                else "not_established"
            ),
            "validates_biology": bool(
                self.living_fly or self.evidence_mode == "recording_reanalysis"
            ),
            "living_fly": self.living_fly,
            "fit_conditions": self.fit_conditions,
            "eval_conditions": self.eval_conditions,
            "contributors": list(self.contributors),
            "results": [r.to_dict() for r in self.results],
            "notes": list(self.notes),
            "created_at": self.created_at,
        }


# Literature-core qualitative weights (same collapse as anatomy.connectome).
# Not MaleCNS synapse counts.
_EDGES: tuple[tuple[str, str, float], ...] = (
    ("R1-R6", "L1", 4.0),
    ("R1-R6", "L2", 4.0),
    ("L1", "T4", 3.0),
    ("L2", "T5", 3.0),
    ("T4", "LPLC2", 3.0),
    ("T5", "LPLC2", 3.0),
    ("LPLC2", "DNp01", 7.0),
    ("LC4", "DNp01", 3.0),
)

_NODES = ("R1-R6", "L1", "L2", "T4", "T5", "LC4", "LPLC2", "DNp01")


def _loom_drive(kind: str, speed: float) -> dict[str, float]:
    """Scalar drives. Expanding loom is the positive class (published direction)."""
    speed = float(speed)
    if kind == "expanding":
        return {"R1-R6": 0.4, "LC4": 0.6 * speed, "LPLC2": 0.9 * speed, "T4": 0.35 * speed, "T5": 0.35 * speed}
    if kind == "receding":
        return {"R1-R6": 0.4, "LC4": 0.15 * speed, "LPLC2": 0.12 * speed, "T4": 0.35 * speed, "T5": 0.35 * speed}
    if kind == "static":
        return {"R1-R6": 0.5, "LC4": 0.05, "LPLC2": 0.05, "T4": 0.05, "T5": 0.05}
    if kind == "luminance":
        return {"R1-R6": 0.9, "LC4": 0.05, "LPLC2": 0.05, "T4": 0.05, "T5": 0.05}
    raise ValueError(kind)


def _simulate(
    kind: str,
    speed: float,
    *,
    silence: str | None = None,
    rescue: bool = False,
    steps: int = 8,
) -> dict[str, float]:
    """Leaky integrator on the literature-core DAG. Excitatory signs assumed."""
    rates = {n: 0.0 for n in _NODES}
    incoming: dict[str, list[tuple[str, float]]] = {n: [] for n in _NODES}
    for a, b, w in _EDGES:
        incoming[b].append((a, w))
    drive = _loom_drive(kind, speed)
    tau = 0.45
    for _ in range(steps):
        nxt = dict(rates)
        for n in _NODES:
            syn = sum(w * rates[src] for src, w in incoming[n]) / 10.0
            ext = drive.get(n, 0.0)
            if silence == n and not rescue:
                ext = 0.0
                syn = 0.0
            elif silence == n and rescue:
                ext = drive.get(n, 0.0)
            nxt[n] = (1.0 - tau) * rates[n] + tau * max(0.0, ext + syn)
        rates = nxt
    return rates


def _selectivity(expanding: float, receding: float, static: float) -> float:
    denom = abs(expanding) + abs(receding) + abs(static) + 1e-6
    return float((expanding - 0.5 * (receding + static)) / denom)


def run_looming_giant_fibre_assay(*, seed: int = 0) -> FlyFunctionalAssay:
    """Reproduce the published *direction* of looming → LPLC2 → DNp01 escape.

    Fit/eval split is by loom speed. No parameters are learned from eval speeds.
    This is a proposed simulation, not a living-fly recording and not a replay
    of Lappalainen et al. 2024 (that implementation is not in this workspace).
    """
    _ = seed
    fit_speeds = (0.8, 1.2)
    eval_speeds = (0.5, 1.6)  # held-out
    stimuli = (
        [("expanding", s, False) for s in fit_speeds]
        + [("receding", s, False) for s in fit_speeds]
        + [("static", 1.0, False)]
        + [("expanding", s, True) for s in eval_speeds]
        + [("receding", s, True) for s in eval_speeds]
        + [("luminance", 1.0, True)]
    )
    results: list[FlyAssayResult] = []
    dnp_expand_eval: list[float] = []
    dnp_recede_eval: list[float] = []
    for kind, speed, held in stimuli:
        rates = _simulate(kind, speed)
        score = _selectivity(
            _simulate("expanding", speed)["DNp01"],
            _simulate("receding", speed)["DNp01"],
            _simulate("static", 1.0)["DNp01"],
        )
        results.append(
            FlyAssayResult(
                stimulus=f"{kind}:{speed}",
                held_out=held,
                rates=rates,
                readout=rates["DNp01"],
                perturbation="none",
                score=score,
                source="proposed_simulation",
            )
        )
        if held and kind == "expanding":
            dnp_expand_eval.append(rates["DNp01"])
        if held and kind == "receding":
            dnp_recede_eval.append(rates["DNp01"])

    # Perturbations on a held-out expanding loom (simulation predictions).
    speed = eval_speeds[0]
    intact = _simulate("expanding", speed)
    silenced = _simulate("expanding", speed, silence="LPLC2")
    rescued = _simulate("expanding", speed, silence="LPLC2", rescue=True)
    random_sil = _simulate("expanding", speed, silence="T4")
    for name, rates, kind in (
        ("silence_LPLC2", silenced, "silence"),
        ("rescue_LPLC2", rescued, "rescue"),
        ("silence_T4_control", random_sil, "random_control"),
    ):
        results.append(
            FlyAssayResult(
                stimulus=f"expanding:{speed}",
                held_out=True,
                rates=rates,
                readout=rates["DNp01"],
                perturbation=name,
                score=(intact["DNp01"] - rates["DNp01"]) / (intact["DNp01"] + 1e-6),
                source="proposed_simulation",
            )
        )
        _ = kind

    drop = intact["DNp01"] - silenced["DNp01"]
    control_drop = intact["DNp01"] - random_sil["DNp01"]
    rescue_ok = rescued["DNp01"] > silenced["DNp01"]
    held_ok = float(np.mean(dnp_expand_eval)) > float(np.mean(dnp_recede_eval))
    selective = drop > control_drop * 1.2
    status = (
        "simulation_reproduces_published_direction_on_held_out_speeds"
        if held_ok and selective and rescue_ok
        else "simulation_failed_published_direction"
    )
    contributors = ["LPLC2"]
    if selective:
        contributors.append("DNp01")
    return FlyFunctionalAssay(
        circuit_id="lplc2_dnp01_looming_escape",
        specimen="literature-core type graph (not a MaleCNS body id dump)",
        dataset_version="literature-core/qualitative-weights; MaleCNS not retrieved",
        evidence_mode="proposed_simulation",
        function_id="feature_discrimination",
        neurons=_NODES,
        citations=(
            "von Reyn et al. 2014 Nat Neurosci 17:962-970 (DNp01/giant fibre escape)",
            "Klapoetke et al. 2017 Nature 543:96-100 (LPLC2 looming)",
            "Namiki et al. 2018 eLife 7:e34272 (descending neurons)",
        ),
        synaptic_sign_assumptions={src + "→" + tgt: "excitatory_assumed" for src, tgt, _ in _EDGES},
        delays_ms={"LPLC2→DNp01": 2.0, "photoreceptor→LPLC2": 8.0},
        stimulus_encoding="expanding vs receding vs static vs luminance scalar loom-speed",
        readout="DNp01 rate (escape-command analogue); not a jump in a living fly",
        validation_status=status,
        living_fly=False,
        fit_conditions=f"qualitative weights fixed; inspected speeds {fit_speeds}",
        eval_conditions=f"held-out speeds {eval_speeds}; silence/rescue on first eval speed",
        contributors=contributors,
        results=results,
        notes=[
            "Passing this assay checks simulator mechanics, not independent biological function.",
            "Not a living-fly experiment. Silencing LPLC2 here is a simulation prediction.",
            "fly-brain vision.statistics is a pixel proxy, not this rate model.",
            "Lappalainen et al. 2024 optic-lobe network is not present in this workspace.",
            "Shared operation with the LLM: detect an approaching-threat description "
            "and raise an action-selection continuation (escape/stop), vs matched controls.",
            f"Held-out expanding DNp01 {float(np.mean(dnp_expand_eval)):.3f} > receding "
            f"{float(np.mean(dnp_recede_eval)):.3f}; LPLC2 silence drop {drop:.3f} vs T4 control {control_drop:.3f}.",
        ],
    )


def assay_signature(assay: FlyFunctionalAssay) -> dict[str, Any]:
    """Functional signature used later for cross-system comparison (not tensor cosine)."""
    if assay.function_id == "value_modulation":
        pam = [r.readout for r in assay.results if "PAM" in r.stimulus]
        ppl = [r.readout for r in assay.results if "PPL1" in r.stimulus]
        cue = [r.readout for r in assay.results if r.stimulus.startswith("cue_only")]
        return {
            "function_id": assay.function_id,
            "preferred_condition": "reward_PAM_gain",
            "rejected_condition": "aversive_PPL1_gain",
            "selectivity": float(np.mean(pam) - np.mean(ppl)) if pam and ppl else 0.0,
            "contributors": list(assay.contributors),
            "perturbation_required_for_causal_claim": True,
            "shared_operation": (
                "A value context scales cue-to-action mapping; mentioning the "
                "modulator by name is not the same operation."
            ),
            "cue_only": float(np.mean(cue)) if cue else None,
        }
    intact = [r for r in assay.results if r.perturbation == "none"]
    expand = [r.readout for r in intact if r.stimulus.startswith("expanding")]
    recede = [r.readout for r in intact if r.stimulus.startswith("receding")]
    return {
        "function_id": assay.function_id,
        "preferred_condition": "expanding_loom",
        "rejected_condition": "receding_or_static",
        "selectivity": float(np.mean(expand) - np.mean(recede)) if expand and recede else 0.0,
        "contributors": list(assay.contributors),
        "perturbation_required_for_causal_claim": True,
        "shared_operation": (
            "Approaching-object evidence increases an action-selection readout; "
            "matched non-approach controls do not."
        ),
    }


def _malecns_w(edges: dict, pre: str, post: str) -> float:
    row = edges.get((pre, post))
    return float(row["weight"]) if row else 0.0


def run_looming_malecns_assay() -> FlyFunctionalAssay:
    """Looming circuit on measured MaleCNS signed synapses + literature stimulus prior."""
    from llmintent.anatomy.flycns import edge_map, load_circuit_bundle

    bundle = load_circuit_bundle()
    edges = edge_map(bundle)
    scale = 1e-5
    w_t4 = _malecns_w(edges, "T4", "LPLC2") * scale
    w_t5 = _malecns_w(edges, "T5", "LPLC2") * scale
    w_lpl = _malecns_w(edges, "LPLC2", "DNp01") * scale
    w_lc4 = _malecns_w(edges, "LC4", "DNp01") * scale

    def rates(kind: str, speed: float, silence: str | None = None, rescue: bool = False) -> dict[str, float]:
        t4 = 0.4 * speed if kind in {"expanding", "receding"} else 0.05
        t5 = 0.4 * speed if kind in {"expanding", "receding"} else 0.05
        # Looming selectivity is a literature drive prior, not in the EM volume.
        lpl_drive = 0.9 * speed if kind == "expanding" else (0.12 * speed if kind == "receding" else 0.05)
        lc4_drive = 0.7 * speed if kind == "expanding" else (0.15 * speed if kind == "receding" else 0.05)
        if silence == "T4" and not rescue:
            t4 = 0.0
        if silence == "LPLC2" and not rescue:
            lpl_drive = 0.0
        lplc2 = max(0.0, lpl_drive + w_t4 * t4 + w_t5 * t5)
        if silence == "LPLC2" and not rescue:
            lplc2 = 0.0
        dnp = max(0.0, w_lpl * lplc2 + w_lc4 * lc4_drive)
        return {"T4": t4, "T5": t5, "LPLC2": lplc2, "LC4": lc4_drive, "DNp01": dnp}

    fit, eval_s = (0.8, 1.2), (0.5, 1.6)
    results: list[FlyAssayResult] = []
    expand_h, recede_h = [], []
    for kind, speed, held in (
        *[("expanding", s, False) for s in fit],
        *[("receding", s, False) for s in fit],
        ("static", 1.0, False),
        *[("expanding", s, True) for s in eval_s],
        *[("receding", s, True) for s in eval_s],
        ("luminance", 1.0, True),
    ):
        r = rates(kind, speed)
        results.append(FlyAssayResult(
            stimulus=f"{kind}:{speed}", held_out=held, rates=r, readout=r["DNp01"],
            perturbation="none", score=r["DNp01"], source="malecns_signed_weights",
        ))
        if held and kind == "expanding":
            expand_h.append(r["DNp01"])
        if held and kind == "receding":
            recede_h.append(r["DNp01"])
    speed = eval_s[0]
    intact = rates("expanding", speed)
    silenced = rates("expanding", speed, silence="LPLC2")
    rescued = rates("expanding", speed, silence="LPLC2", rescue=True)
    t4c = rates("expanding", speed, silence="T4")
    for name, r in (("silence_LPLC2", silenced), ("rescue_LPLC2", rescued), ("silence_T4_control", t4c)):
        results.append(FlyAssayResult(
            stimulus=f"expanding:{speed}", held_out=True, rates=r, readout=r["DNp01"],
            perturbation=name, score=(intact["DNp01"] - r["DNp01"]) / (intact["DNp01"] + 1e-9),
            source="malecns_signed_weights",
        ))
    drop = intact["DNp01"] - silenced["DNp01"]
    ctrl = intact["DNp01"] - t4c["DNp01"]
    held_ok = float(np.mean(expand_h)) > float(np.mean(recede_h))
    selective = drop > ctrl
    status = (
        "malecns_weights_reproduce_published_direction_on_held_out_speeds"
        if held_ok and selective and rescued["DNp01"] > silenced["DNp01"]
        else "malecns_weights_failed_published_direction"
    )
    return FlyFunctionalAssay(
        circuit_id="malecns_lplc2_dnp01_looming_escape",
        specimen="MaleCNS v1.0 adult male, traced-only (extracted circuit)",
        dataset_version=str(bundle.get("edge_set")),
        evidence_mode="proposed_simulation",
        function_id="feature_discrimination",
        neurons=("T4", "T5", "LPLC2", "LC4", "DNp01"),
        citations=(
            "MaleCNS v1.0 traced weights + consensus_nt (Janelia/Google 2026)",
            "Klapoetke et al. 2017 Nature 543:96-100 (LPLC2 looming selectivity, not in EM)",
            "von Reyn et al. 2014 Nat Neurosci 17:962-970 (DNp01 escape)",
        ),
        synaptic_sign_assumptions={"LPLC2→DNp01": "acetylcholine_excitatory_measured"},
        delays_ms={"LPLC2→DNp01": 2.0},
        stimulus_encoding="literature prior on expanding vs receding; weights are MaleCNS",
        readout="DNp01 rate from signed MaleCNS synapses",
        validation_status=status,
        living_fly=False,
        fit_conditions=f"inspected speeds {fit}",
        eval_conditions=f"held-out speeds {eval_s}",
        contributors=["LPLC2", "DNp01"] if selective else ["LPLC2"],
        results=results,
        notes=[
            "Passing this assay checks simulator mechanics, not independent biological function.",
            "Weights and NTs are from the downloaded fly-brain MaleCNS tables.",
            "Expanding vs receding drive is still a literature prior; the EM volume has no spikes.",
            "Not a living-fly experiment. LPLC2 silence is a simulation on measured synapses.",
            f"LPLC2→DNp01 {edges[('LPLC2','DNp01')]['weight']} synapses, all ACh; "
            f"LC4→DNp01 {edges[('LC4','DNp01')]['weight']}.",
        ],
    )


def run_dopamine_malecns_assay() -> FlyFunctionalAssay:
    """PAM vs PPL1 dopamine as gain on KC→MBON, not as extra acetylcholine current."""
    from llmintent.anatomy.flycns import edge_map, load_circuit_bundle

    bundle = load_circuit_bundle()
    edges = edge_map(bundle)
    scale = 1e-6
    w_kc_mbon = abs(_malecns_w(edges, "KC", "MBON")) * scale
    w_apl = abs(_malecns_w(edges, "APL", "KC")) * scale

    def step(cue: float, pam: float, ppl: float, extra_ach: float, apl_on: bool) -> dict[str, float]:
        inh = w_apl if apl_on else 0.0
        kc = max(0.0, cue + extra_ach - 0.15 * inh)
        gain = 1.0 + pam - ppl
        mbon = max(0.0, w_kc_mbon * kc * gain)
        return {"KC": kc, "MBON": mbon, "PAM_gain": pam, "PPL1_gain": ppl, "extra_ACh": extra_ach}

    cue_fit, cue_eval = 1.0, 0.7
    conds = [
        ("cue_only", cue_fit, False, dict(cue=cue_fit, pam=0.0, ppl=0.0, extra_ach=0.0, apl_on=True)),
        ("cue_PAM", cue_fit, False, dict(cue=cue_fit, pam=0.5, ppl=0.0, extra_ach=0.0, apl_on=True)),
        ("cue_PPL1", cue_fit, False, dict(cue=cue_fit, pam=0.0, ppl=0.5, extra_ach=0.0, apl_on=True)),
        ("cue_extra_ACh", cue_fit, False, dict(cue=cue_fit, pam=0.0, ppl=0.0, extra_ach=0.5, apl_on=True)),
        ("cue_only_held", cue_eval, True, dict(cue=cue_eval, pam=0.0, ppl=0.0, extra_ach=0.0, apl_on=True)),
        ("cue_PAM_held", cue_eval, True, dict(cue=cue_eval, pam=0.5, ppl=0.0, extra_ach=0.0, apl_on=True)),
        ("cue_PPL1_held", cue_eval, True, dict(cue=cue_eval, pam=0.0, ppl=0.5, extra_ach=0.0, apl_on=True)),
        ("cue_extra_ACh_held", cue_eval, True, dict(cue=cue_eval, pam=0.0, ppl=0.0, extra_ach=0.5, apl_on=True)),
        ("silence_APL", cue_eval, True, dict(cue=cue_eval, pam=0.0, ppl=0.0, extra_ach=0.0, apl_on=False)),
    ]
    results = []
    held_rows = {}
    for name, cue, held, kw in conds:
        r = step(**kw)
        results.append(FlyAssayResult(
            stimulus=name, held_out=held, rates=r, readout=r["MBON"],
            perturbation=name, score=r["MBON"] / (r["KC"] + 1e-9),
            source="malecns_signed_weights",
        ))
        if held:
            held_rows[name] = r
    pam_up = held_rows["cue_PAM_held"]["MBON"] > held_rows["cue_only_held"]["MBON"]
    ppl_down = held_rows["cue_PPL1_held"]["MBON"] < held_rows["cue_only_held"]["MBON"]
    ratio_pam = held_rows["cue_PAM_held"]["MBON"] / (held_rows["cue_PAM_held"]["KC"] + 1e-9)
    ratio_ach = held_rows["cue_extra_ACh_held"]["MBON"] / (held_rows["cue_extra_ACh_held"]["KC"] + 1e-9)
    ratio_cue = held_rows["cue_only_held"]["MBON"] / (held_rows["cue_only_held"]["KC"] + 1e-9)
    gain_not_current = ratio_pam > ratio_ach and abs(ratio_ach - ratio_cue) < abs(ratio_pam - ratio_cue)
    status = (
        "malecns_dopamine_gain_dissociates_from_acetylcholine_current"
        if pam_up and ppl_down and gain_not_current
        else "malecns_dopamine_assay_failed_dissociation"
    )
    return FlyFunctionalAssay(
        circuit_id="malecns_pam_ppl1_kc_mbon",
        specimen="MaleCNS v1.0 adult male, traced-only (extracted circuit)",
        dataset_version=str(bundle.get("edge_set")),
        evidence_mode="proposed_simulation",
        function_id="value_modulation",
        neurons=("PAM", "PPL1", "KC", "MBON", "APL"),
        citations=(
            "MaleCNS v1.0: PAM 316/316 dopamine; PPL1 16/16 dopamine; APL 2/2 GABA",
            "Aso et al. 2014 eLife 3:e04577 (MB compartments / DAN-MBON)",
            "Ichinose et al. 2015; reward vs aversive DA onto KC is a literature prior on function",
        ),
        synaptic_sign_assumptions={
            "PAM→KC": "dopamine_modulatory_not_fast_ACh",
            "PPL1→KC": "dopamine_modulatory_not_fast_ACh",
            "APL→KC": "gaba_inhibitory_measured_negative_weight",
            "KC→MBON": "acetylcholine_majority_excitatory",
        },
        delays_ms={"PAM→KC": 20.0},
        stimulus_encoding="cue current into KC; PAM/PPL1 are gain on KC→MBON",
        readout="MBON rate and MBON/KC ratio",
        validation_status=status,
        living_fly=False,
        fit_conditions="cue=1.0 inspected",
        eval_conditions="held-out cue=0.7",
        contributors=["PAM", "PPL1"] if pam_up and ppl_down else [],
        results=results,
        notes=[
            "Passing this assay checks simulator mechanics (gain vs extra ACh), not independent biological function.",
            "PAM and PPL1 are dopamine on MaleCNS. They are not modeled as extra excitatory synapses.",
            "GABA APL→KC is the signed inhibitory control (weight negative in the traced graph).",
            "Shared LLM operation: a value context should scale cue→outcome without being the cue, "
            "and without being the word dopamine.",
        ],
    )
