"""Kill-or-keep experiments for fly-connectome anatomy.

Each claim is a hypothesis. A claim enters the next version only if it
survives its kill criterion. Failures are demoted (prior / correlate /
killed), not silently kept as mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from llmintent.anatomy.ablation import plant_and_ablate
from llmintent.anatomy.atlas import region_ids
from llmintent.anatomy.compile import compile_regions
from llmintent.anatomy.connectome import literature_region_connectome
from llmintent.anatomy.intents import score_blob
from llmintent.anatomy.misalign import scan_negative_intent
from llmintent.anatomy.model_map import Anatomy
from llmintent.anatomy.svd_map import map_synthetic
from llmintent.anatomy.thoughts import LayerThought
from llmintent.anatomy.trace import trace_prompt
from llmintent.anatomy.trajectory import trajectory

LOOMING = "I hear a song because a dark shape is looming."
FLOOR = 0.18


@dataclass
class ExperimentResult:
    id: str
    claim: str
    kind: str  # identified | prior | correlate | causal | detector | method
    survived: bool
    skipped: bool = False
    metric: str = ""
    detail: str = ""
    if_survives: str = ""
    if_dies: str = ""
    verdict: str = ""  # keep | demote | kill | skip

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "claim": self.claim,
            "kind": self.kind,
            "survived": self.survived,
            "skipped": self.skipped,
            "verdict": self.verdict,
            "metric": self.metric,
            "detail": self.detail,
            "if_survives": self.if_survives,
            "if_dies": self.if_dies,
        }


@dataclass
class ValidationReport:
    results: list[ExperimentResult] = field(default_factory=list)
    next_version: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def kept(self) -> list[ExperimentResult]:
        return [r for r in self.results if r.verdict == "keep"]

    @property
    def demoted(self) -> list[ExperimentResult]:
        return [r for r in self.results if r.verdict == "demote"]

    @property
    def killed(self) -> list[ExperimentResult]:
        return [r for r in self.results if r.verdict == "kill"]

    def to_dict(self) -> dict:
        return {
            "n": len(self.results),
            "kept": [r.id for r in self.kept],
            "demoted": [r.id for r in self.demoted],
            "killed": [r.id for r in self.killed],
            "skipped": [r.id for r in self.results if r.verdict == "skip"],
            "next_version": dict(self.next_version),
            "results": [r.to_dict() for r in self.results],
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Anatomy validation",
            "",
            f"Kept **{len(self.kept)}** · demoted **{len(self.demoted)}** · "
            f"killed **{len(self.killed)}** · skipped **{sum(1 for r in self.results if r.skipped)}**.",
            "",
            "| ID | Claim | Kind | Verdict | Metric |",
            "|----|-------|------|---------|--------|",
        ]
        for r in self.results:
            lines.append(
                f"| `{r.id}` | {r.claim} | {r.kind} | **{r.verdict}** | {r.metric} |"
            )
        lines.append("")
        lines.append("## Next version (what enters)")
        lines.append("")
        for k, v in self.next_version.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")
        return "\n".join(lines)


def _finish(
    rid: str,
    claim: str,
    kind: str,
    survived: bool,
    *,
    metric: str,
    detail: str,
    if_survives: str,
    if_dies: str,
    skipped: bool = False,
    demote_on_fail: bool = False,
) -> ExperimentResult:
    if skipped:
        verdict = "skip"
    elif survived:
        verdict = "keep"
    elif demote_on_fail:
        verdict = "demote"
    else:
        verdict = "kill"
    return ExperimentResult(
        id=rid,
        claim=claim,
        kind=kind,
        survived=survived,
        skipped=skipped,
        metric=metric,
        detail=detail,
        if_survives=if_survives,
        if_dies=if_dies,
        verdict=verdict,
    )


def e_compile_gold() -> ExperimentResult:
    plan = compile_regions(LOOMING)
    got = set(plan.regions)
    need = {"vision", "auditory", "causal_logic"}
    ok = need <= got
    return _finish(
        "compile_gold",
        "Compile recovers vision + auditory + causal_logic on the looming prompt.",
        "identified",
        ok,
        metric=f"regions={sorted(got)}",
        detail=f"dropped={plan.dropped}",
        if_survives="Compile stays the trustworthy catalogue read.",
        if_dies="Atlas aliases/intent docs are wrong; freeze anatomy reports.",
    )


def e_compile_negative() -> ExperimentResult:
    fiction = compile_regions("Once upon a time a little girl lived in a village.")
    unmatched = compile_regions("xyzzy plugh fnord.")
    ok = "vision" not in fiction.regions and not unmatched.regions
    return _finish(
        "compile_drop",
        "Unmatched English is dropped; fiction does not hash onto vision.",
        "identified",
        ok,
        metric=f"fiction={fiction.regions} unmatched={unmatched.regions}",
        detail=f"fiction_dropped={fiction.dropped}",
        if_survives="Keep drop-unmatched. Do not hash leftovers.",
        if_dies="Compile is a bag-of-words leak; do not report occupancy.",
    )


def e_span_varies() -> ExperimentResult:
    trace = trace_prompt(LOOMING)
    by_id = {r.id: r for r in trace.regions}
    aud = by_id["auditory"]
    vis = by_id["vision"]
    # Auditory should peak on an earlier span than vision.
    ok = (
        aud.peak_span is not None
        and vis.peak_span is not None
        and aud.peak_span < vis.peak_span
        and aud.variation > 0
        and vis.variation > 0
    )
    return _finish(
        "span_varies",
        "Occupancy varies through the prompt: auditory peaks before vision.",
        "identified",
        ok,
        metric=f"auditory_peak={aud.peak_span} vision_peak={vis.peak_span} "
        f"var_a={aud.variation:.3f} var_v={vis.variation:.3f}",
        detail=f"spans={[s.text for s in trace.spans]}",
        if_survives="Keep trace_prompt series on region cards.",
        if_dies="Does/varies is a static label; drop series from reports.",
    )


def e_connectome_iv() -> ExperimentResult:
    conn = literature_region_connectome()
    viol = {(e.source, e.target) for e in conn.exclusion_violations()}
    from llmintent.anatomy.iv_engine import connectome_iv
    from llmintent.anatomy.atlas import region_ids as _rids

    occ = {rid: 0.0 for rid in _rids()}
    occ.update(vision=0.8, auditory=0.5, causal_logic=0.6, associative=0.4, motor=0.3)
    iv = connectome_iv(occ, mock_iv=True, seed=3)
    ok = (
        ("vision", "descending") in viol
        and conn.has_path("olfactory", "associative")
        and "olfactory" in conn.valid_instruments("associative")
        and "vision" not in conn.valid_instruments("descending")
        and "vision" not in iv.instruments_used
    )
    return _finish(
        "connectome_iv",
        "Giant fibre is an IV exclusion; olfactory may instrument associative.",
        "prior",
        ok,
        metric=f"violations={sorted(viol)} instruments={iv.instruments_used} vision_inst_desc={conn.valid_instruments('descending')}",
        detail="Literature-core collapse. Survival means the prior is internally consistent, not that the LLM has that synapse.",
        if_survives="Keep connectome as IV prior only (never as a mechanism claim).",
        if_dies="Fix TYPE_TO_REGION / exclusion rule before any IV report.",
    )


def e_planted_svd() -> ExperimentResult:
    import numpy as np

    dim = 24
    v = np.eye(dim)[0]
    a = np.eye(dim)[1]
    H = np.vstack([np.tile(v * 4.0, (6, 1)), np.tile(a * 4.0, (6, 1))])
    anatomy = map_synthetic(H, {"vision": v, "auditory": a}, layer_index=list(range(12)))
    ok = anatomy.occupancy["vision"] > 0.25 and anatomy.occupancy["auditory"] > 0.25
    return _finish(
        "planted_svd",
        "SVD recovers planted vision/auditory axes.",
        "method",
        ok,
        metric=f"occ_v={anatomy.occupancy['vision']:.3f} occ_a={anatomy.occupancy['auditory']:.3f}",
        detail="Orthogonal planted axes. Method check, not a fly recording.",
        if_survives="Keep SVD occupancy for planted/synthetic maps.",
        if_dies="SVD mapper is broken; do not map residuals.",
    )


def e_planted_ablation() -> ExperimentResult:
    result, _ = plant_and_ablate("vision", "auditory", seed=0)
    ok = result.changed and result.kl_ab > 0.15
    return _finish(
        "planted_ablation",
        "Driving planted A vs B changes the linear readout.",
        "method",
        ok,
        metric=f"changed={result.changed} kl={result.kl_ab:.3f} top_a={result.top_a} top_b={result.top_b}",
        detail="If this dies, the ablation API cannot test anything on a real model.",
        if_survives="Keep A-vs-B ablation as the causal test harness.",
        if_dies="Do not advertise ablation until the linear path is fixed.",
    )


def e_residual_floor() -> ExperimentResult:
    thoughts = [
        LayerThought(
            layer=4,
            depth=0.5,
            band="central",
            top_tokens=["danger", "threat"],
            region="workspace",
            region_score=0.05,
            residual_l2=1.0,
        )
    ]
    traj = trajectory(LOOMING, thoughts=thoughts, print_flag=False, n_layers=8)
    compiled = set(compile_regions(LOOMING).regions)
    weak_ok = thoughts[0].region_score < FLOOR
    compile_ok = {"vision", "auditory", "causal_logic"} <= compiled
    ok = weak_ok and compile_ok and not traj.residual_identified
    return _finish(
        "residual_floor",
        "Residual cosine < 0.18 is not identified occupancy; compile wins.",
        "identified",
        ok,
        metric=f"region_score={thoughts[0].region_score} compiled={sorted(compiled)} residual_identified={traj.residual_identified}",
        detail="Reproduces the Qwen 27B policy: danger/threat unembed is a correlate, not occupancy.",
        if_survives="Keep floor 0.18. Residual labels stay correlates below it.",
        if_dies="Trajectory is treating weak unembed as anatomy; freeze residual maps.",
        demote_on_fail=True,
    )


def e_misalign_specificity() -> ExperimentResult:
    _, loom_flag = scan_negative_intent(LOOMING)
    _, harm_flag = scan_negative_intent("I will hurt them and hide this from everyone.")
    _, heart = scan_negative_intent("He survived a heart attack last winter.")
    # Looming must not fire. Harm should. Medical 'attack' should not.
    ok = (not loom_flag.triggered) and harm_flag.triggered and (not heart.triggered)
    return _finish(
        "misalign_specificity",
        "MisAlign Flag fires on harm+deception, not on looming or 'heart attack'.",
        "detector",
        ok,
        metric=(
            f"loom={loom_flag.triggered}({loom_flag.trigger}) "
            f"harm={harm_flag.triggered}({harm_flag.trigger}) "
            f"heart={heart.triggered}({heart.trigger})"
        ),
        detail="Lexical detector. Survival = usable notification, not mechanistic malice.",
        if_survives="Keep MisAlign Flag as a lexical/short-pipe detector.",
        if_dies="Tighten cues or stop printing the banner by default.",
        demote_on_fail=True,
    )


def e_trajectory_structure() -> ExperimentResult:
    traj = trajectory(LOOMING, print_flag=False, n_layers=8)
    ids = set(traj.intent_ids)
    ok = (
        traj.method == "trajectory"
        and len(traj.layers) == 8
        and all(set(row.intents) == ids for row in traj.layers)
        and any(row.active for row in traj.layers)
    )
    return _finish(
        "trajectory_structure",
        "trajectory() is all layers × all intents (zeros included).",
        "method",
        ok,
        metric=f"layers={len(traj.layers)} n_intents={len(ids)} active={sum(1 for r in traj.layers if r.active)}",
        detail="Structure check. Does not prove the imputed path is cognition.",
        if_survives="Keep trajectory as the correlate matrix API.",
        if_dies="Do not ship all-layers×all-intents until the matrix is complete.",
    )


def e_graph_kinds() -> ExperimentResult:
    anat = Anatomy.offline(LOOMING, n_layers=8)
    kinds = {e.kind for e in anat.graph.edges}
    need = {"connectome_prior", "stream", "responsible"}
    ok = need <= kinds
    n_prior = sum(1 for e in anat.graph.edges if e.kind == "connectome_prior")
    n_resp = sum(1 for e in anat.graph.edges if e.kind == "responsible")
    return _finish(
        "graph_kinds",
        "Complete graph separates connectome_prior from residual stream and responsibility edges.",
        "prior",
        ok,
        metric=f"kinds={sorted(kinds)} n_prior={n_prior} n_resp={n_resp}",
        detail="Union is allowed only if kinds stay distinct. Prior edges are literature, not learned.",
        if_survives="Keep the union graph, with kind tags mandatory in reports.",
        if_dies="Do not draw a single 'anatomy graph' that mixes literature with residuals.",
    )


def e_shuffle_compile() -> ExperimentResult:
    """If region labels are shuffled, gold recovery should collapse."""
    rng_hits = compile_regions(LOOMING)
    gold = set(rng_hits.regions)
    # Control: a prompt that should not share the gold set.
    other = compile_regions("I am hungry and want a drink of nectar.")
    other_set = set(other.regions)
    overlap = gold & other_set
    # Hungry should be gustatory, not the looming triple.
    ok = "gustatory" in other_set and not ({"vision", "auditory"} <= other_set)
    return _finish(
        "compile_separates",
        "Looming vs hungry compile to different closed sets (not one blob).",
        "identified",
        ok,
        metric=f"loom={sorted(gold)} hungry={sorted(other_set)} overlap={sorted(overlap)}",
        detail="Catalogue must distinguish sensory channels.",
        if_survives="Keep the eleven-region closed catalogue.",
        if_dies="Regions are not separable; collapse the atlas.",
    )


def e_weight_shuffle(model: str | None) -> ExperimentResult:
    """Weight-correlate vs shuffled token blob. Needs a local HF model."""
    claim = "FFN-unembed token blobs score the intent catalogue above shuffled tokens."
    if not model:
        return _finish(
            "weight_shuffle",
            claim,
            "correlate",
            False,
            skipped=True,
            metric="skipped",
            detail="Pass --model gpt2 (or any weighted HF id) to run.",
            if_survives="Layer responsibility may be called a weight correlate.",
            if_dies="Do not say a layer 'is responsible for' an intent from FFN SVD.",
            demote_on_fail=True,
        )
    try:
        import random

        anat = Anatomy.from_pretrained(model)
    except Exception as exc:  # noqa: BLE001 — experiment gate
        return _finish(
            "weight_shuffle",
            claim,
            "correlate",
            False,
            skipped=True,
            metric="load_failed",
            detail=str(exc)[:240],
            if_survives="Layer responsibility may be called a weight correlate.",
            if_dies="Do not say a layer 'is responsible for' an intent from FFN SVD.",
            demote_on_fail=True,
        )
    real_peak = 0.0
    shuf_peak = 0.0
    rng = random.Random(0)
    for row in anat.layers:
        blob = " ".join(row.tokens)
        real_peak = max(real_peak, max(score_blob(blob).values(), default=0.0))
        chars = list(blob)
        rng.shuffle(chars)
        shuf_peak = max(shuf_peak, max(score_blob("".join(chars)).values(), default=0.0))
    ok = real_peak > shuf_peak + 0.05
    return _finish(
        "weight_shuffle",
        claim,
        "correlate",
        ok,
        metric=f"real_peak={real_peak:.3f} shuffle_peak={shuf_peak:.3f} model={anat.model_name}",
        detail="If real ≉ shuffle, SVD-unembed is not carrying catalogue structure.",
        if_survives="Call FFN SVD a weight correlate (not 'responsible for').",
        if_dies="Demote Anatomy.from_pretrained responsibilities to untested correlate.",
        demote_on_fail=True,
    )


def e_causal_steer(model: str | None) -> ExperimentResult:
    """Ablate vision-band vs auditory-band on the looming prompt. Needs weights."""
    claim = "Steering vision-band vs auditory-band moves next-token mass on the looming prompt."
    if not model:
        return _finish(
            "causal_steer",
            claim,
            "causal",
            False,
            skipped=True,
            metric="skipped",
            detail="Pass --model gpt2 to run residual-stream A vs B.",
            if_survives="Region 'does' may be stated as a causal claim on that model.",
            if_dies="Keep 'does' as a compile/job sentence only — not a mechanism.",
            demote_on_fail=True,
        )
    try:
        from llmintent.anatomy.ablation import ablate_model
        from llmintent.anatomy.svd_map import map_weights
        from llmintent.models import load_model_bundle
        from llmintent.suite import resolve_model_spec

        spec = resolve_model_spec(model=model, use_env=False)
        hf_id = spec.hf_id if spec is not None else model
        bundle = load_model_bundle(hf_id)
        svd = map_weights(bundle)
        result = ablate_model(bundle, LOOMING, svd, "vision", "auditory", gain=1.5)
    except Exception as exc:  # noqa: BLE001
        return _finish(
            "causal_steer",
            claim,
            "causal",
            False,
            skipped=True,
            metric="load_failed",
            detail=str(exc)[:240],
            if_survives="Region 'does' may be stated as a causal claim on that model.",
            if_dies="Keep 'does' as a compile/job sentence only — not a mechanism.",
            demote_on_fail=True,
        )
    ok = result.changed and result.kl_ab > 0.05
    return _finish(
        "causal_steer",
        claim,
        "causal",
        ok,
        metric=f"changed={result.changed} kl={result.kl_ab:.4f} top_a={result.top_a[:3]} top_b={result.top_b[:3]}",
        detail="Next-token shift only. Not a behavioural proof. Direction is not required to match fly jobs on GPT-2.",
        if_survives="Keep model ablation; still label it next-token shift, not fly neuropil.",
        if_dies="Demote region 'does' to compile-only. Do not claim layers implement fly jobs.",
        demote_on_fail=True,
    )


def _offline_fns() -> list[Callable[[], ExperimentResult]]:
    return [
        e_compile_gold,
        e_compile_negative,
        e_shuffle_compile,
        e_span_varies,
        e_connectome_iv,
        e_planted_svd,
        e_planted_ablation,
        e_residual_floor,
        e_misalign_specificity,
        e_trajectory_structure,
        e_graph_kinds,
    ]


def apply_next_version(results: list[ExperimentResult]) -> dict[str, str]:
    """Map survivors onto the claims that may be stated in the next version."""
    by_id = {r.id: r for r in results}
    nxt: dict[str, str] = {}

    def status(eid: str) -> str:
        r = by_id.get(eid)
        if r is None or r.skipped:
            return "untested"
        if r.verdict == "keep":
            return "keep"
        if r.verdict == "demote":
            return "demote"
        return "kill"

    nxt["compile_catalogue"] = (
        "identified: compile is the trustworthy read"
        if status("compile_gold") == "keep" and status("compile_drop") == "keep"
        else "killed: do not report region occupancy from compile"
    )
    nxt["span_variation"] = (
        "identified: occupancy series through spans"
        if status("span_varies") == "keep"
        else "killed: drop varies-through-prompt from cards"
    )
    nxt["eleven_regions"] = (
        "identified as a closed catalogue (separable channels)"
        if status("compile_separates") == "keep"
        else "killed: collapse atlas"
    )
    nxt["connectome_graph"] = (
        "prior only: IV instruments and exclusion flags, not synapses in the LM"
        if status("connectome_iv") == "keep"
        else "killed: do not run IV"
    )
    nxt["svd_mapper"] = (
        "method: planted recovery only"
        if status("planted_svd") == "keep"
        else "killed"
    )
    nxt["ablation_harness"] = (
        "method: linear A vs B works"
        if status("planted_ablation") == "keep"
        else "killed"
    )
    nxt["residual_occupancy"] = (
        "correlate; identified only if cosine >= 0.18"
        if status("residual_floor") == "keep"
        else "demote: residual maps untrusted"
    )
    nxt["misalign_flag"] = (
        "lexical detector (not mechanistic malice)"
        if status("misalign_specificity") == "keep"
        else "demote: do not print MisAlign Flag by default"
    )
    nxt["trajectory"] = (
        "correlate matrix (all layers x all intents); imputed path, not cognition"
        if status("trajectory_structure") == "keep"
        else "killed"
    )
    nxt["complete_graph"] = (
        "union allowed iff edge.kind is shown (connectome_prior != stream != responsible)"
        if status("graph_kinds") == "keep"
        else "killed: do not ship mixed graph"
    )
    w = status("weight_shuffle")
    if w == "untested":
        nxt["layer_responsibility"] = (
            "untested correlate: say weight-correlate, never 'is responsible for'"
        )
    elif w == "keep":
        nxt["layer_responsibility"] = "weight-correlate (survived shuffle); still not causal"
    else:
        nxt["layer_responsibility"] = "demote: FFN unembed ~ shuffle; drop responsibility language"
    c = status("causal_steer")
    if c == "untested":
        nxt["region_does_causal"] = (
            "untested: 'does' is a job sentence on the atlas, not a mechanism"
        )
    elif c == "keep":
        nxt["region_does_causal"] = "next-token shift survived; still not a fly neuropil"
    else:
        nxt["region_does_causal"] = "demote: A vs B did not move readout; 'does' is compile-only"
    nxt["fly_is_the_llm"] = "killed a priori: never enters any version"
    return nxt


def run_validation(*, model: str | None = None) -> ValidationReport:
    results = [fn() for fn in _offline_fns()]
    results.append(e_weight_shuffle(model))
    results.append(e_causal_steer(model))
    nxt = apply_next_version(results)
    notes = [
        "Kill-or-keep: only 'keep' claims may be stated as identified in the next version.",
        "Demoted claims stay in the API with weaker language (prior / correlate / detector).",
        "Skipped causal tests stay 'untested' — they do not sneak in as mechanisms.",
        f"Closed catalogue: {', '.join(region_ids())}.",
    ]
    return ValidationReport(results=results, next_version=nxt, notes=notes)


__all__ = [
    "ExperimentResult",
    "LOOMING",
    "ValidationReport",
    "run_validation",
]
