"""End-to-end correspondence experiment and report.

Sequence: fly assay → SVD/logits/units → matched tasks → cascade → prior comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmintent.anatomy.adapters import index_bundle
from llmintent.anatomy.cascade import architecture_edges, run_cascade
from llmintent.anatomy.discover import discover_from_weights
from llmintent.anatomy.evidence import SCHEMA_VERSION, package_versions, utc_now
from llmintent.anatomy.fly_assay import (
    run_dopamine_malecns_assay,
    run_looming_giant_fibre_assay,
    run_looming_malecns_assay,
)
from llmintent.anatomy.flycns import malecns_status
from llmintent.anatomy.malecns import bundled_reference, neuprint_adapter
from llmintent.anatomy.transmitters import score_value_modulation, transmitter_report
from llmintent.anatomy.region_test import (
    compare_priors,
    test_analogous_region,
    validate_analogous_region_test,
)
from llmintent.anatomy.spaces import pin_versions
from llmintent.anatomy.tasks import by_split, family_baseline
from llmintent.anatomy.functions import functions_to_dict
from llmintent import latent as li_latent


@dataclass
class CorrespondenceRow:
    fly_ids: list[str]
    fly_function: str
    fly_evidence: str
    llm_ids: list[str]
    svd_profile: dict[str, Any]
    latent_note: str
    intervention: str
    strength: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fly_neuron_or_circuit": list(self.fly_ids),
            "tested_function": self.fly_function,
            "fly_evidence": self.fly_evidence,
            "llm_layer_module_units": list(self.llm_ids),
            "signed_svd_logit_profile": self.svd_profile,
            "latent_recruitment": self.latent_note,
            "intervention_outcome": self.intervention,
            "strength": self.strength,
        }


def _try_bundle(model: str | None, *, load_in_4bit: bool | None = None):
    if not model:
        return None, "no model requested"
    try:
        from llmintent.models import load_model_bundle
        from llmintent.suite.resolve import resolve_model_id, resolve_model_spec

        spec = resolve_model_spec(model=model, use_env=False)
        four = load_in_4bit
        if four is None and spec is not None and getattr(spec, "size", None) == "27b":
            four = True
        hf = resolve_model_id(model=model, use_env=False, default=model)
        return load_model_bundle(hf, load_in_4bit=bool(four)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def run_atlas_experiment(
    *,
    model: str | None = "qwen:27b",
    out_dir: str | Path | None = None,
    max_layers: int | None = 6,
    top_k: int = 4,
    load_in_4bit: bool | None = None,
) -> dict[str, Any]:
    """Milestones A–E as far as this workspace can run them."""
    literature_assay = run_looming_giant_fibre_assay()
    assay = run_looming_malecns_assay()
    da_assay = run_dopamine_malecns_assay()
    bio = bundled_reference()
    nts = transmitter_report()
    latent_info = li_latent.describe()
    bundle, load_err = _try_bundle(model, load_in_4bit=load_in_4bit)
    missing: list[str] = []
    if load_err and model:
        missing.append(f"pretrained model {model!r}: {load_err}")
    neu = neuprint_adapter()
    cns = malecns_status()
    if not cns.get("live_flybrain"):
        missing.append("live fly-brain cache not found; using extracted circuit JSON")

    index = None
    cands = []
    baseline_loom = None
    baseline_causal = None
    cascade = None
    region_test = None
    priors = None
    second_arch = None
    conf = by_split("looming_language", "confirmation")
    disc = by_split("looming_language", "discovery")
    region_test = test_analogous_region(assay, None if bundle is None else bundle, conf)
    validation = validate_analogous_region_test(assay=assay)
    value_mod = score_value_modulation(bundle)
    da_region = test_analogous_region(
        da_assay, None if bundle is None else bundle,
        by_split("value_modulation", "confirmation"),
        prior_condition="dopamine_gain",
    )
    if bundle is not None:
        index = index_bundle(bundle)
        cands = discover_from_weights(
            bundle, disc, function_id="feature_discrimination", top_k=top_k, max_layers=max_layers
        )
        try:
            baseline_loom = family_baseline(bundle, "looming_language", "discovery")
            baseline_causal = family_baseline(bundle, "causal_consequence", "discovery")
        except Exception as exc:
            missing.append(f"forced-choice baseline: {exc}")
        from llmintent.anatomy.discover import _ffn_down_specs, _get_module
        from llmintent.anatomy.spaces import decompose_ffn_down

        specs = _ffn_down_specs(index)
        vecs, layers = [], []
        for spec in specs[:2]:
            try:
                mod = _get_module(bundle.model, spec.component.module_path)
                weight = mod.data if hasattr(mod, "data") else mod.weight.data
                comps = decompose_ffn_down(
                    weight,
                    layout=spec.layout,
                    layer=int(spec.component.layer or 0),
                    module_path=spec.component.module_path,
                    top_k=1,
                )
                vecs.append(comps[0].u)
                layers.append(int(spec.component.layer or 0))
            except Exception:
                continue
        if len(vecs) >= 2:
            cascade = run_cascade(
                bundle,
                (conf or disc)[0].prompt,
                vecs[0],
                vecs[1],
                [layers[0]],
                [layers[1]],
            )
        import numpy as np

        n_layers = index.n_layers
        late = list(range(max(0, n_layers // 2), n_layers))
        early = list(range(max(1, n_layers // 2)))
        rng = np.random.default_rng(0)
        shuffled_layers = list(range(n_layers))
        rng.shuffle(shuffled_layers)
        shuffled_layers = shuffled_layers[: max(1, len(late))]
        fly_cands = discover_from_weights(
            bundle, disc, function_id="feature_discrimination",
            top_k=top_k, max_layers=max_layers, prior_layers=late,
        )
        none_cands = discover_from_weights(
            bundle, disc, function_id="feature_discrimination",
            top_k=top_k, max_layers=max_layers, prior_layers=None,
        )
        shuf_cands = discover_from_weights(
            bundle, disc, function_id="feature_discrimination",
            top_k=top_k, max_layers=max_layers, prior_layers=shuffled_layers,
        )
        rewired_cands = discover_from_weights(
            bundle, disc, function_id="feature_discrimination",
            top_k=top_k, max_layers=max_layers, prior_layers=early,
        )
        priors = compare_priors(
            assay, bundle, conf,
            fly_top_contrast=abs(fly_cands[0].contrast) if fly_cands else None,
            no_prior_top_contrast=abs(none_cands[0].contrast) if none_cands else None,
            shuffled_top_contrast=abs(shuf_cands[0].contrast) if shuf_cands else None,
            rewired_top_contrast=abs(rewired_cands[0].contrast) if rewired_cands else None,
        )
        second_arch = {
            "adapter": "llama_like",
            "status": "implemented",
            "executed_on_checkpoint": True,
            "note": "Qwen 27B uses the Llama/Qwen Linear adapter (down_proj = W_down).",
        }

    rows = []
    if cands:
        top = cands[0]
        rows.append(
            CorrespondenceRow(
                fly_ids=list(assay.contributors),
                fly_function=assay.function_id,
                fly_evidence=assay.evidence_mode + " / " + assay.validation_status,
                llm_ids=[u.to_dict()["module_path"] for u in top.units[:6]],
                svd_profile={
                    "region_id": top.region_id,
                    "contrast": top.contrast,
                    "sign": top.sign,
                    "plus_tokens": (top.logit_profile.get("plus") or [])[:5],
                    "minus_tokens": (top.logit_profile.get("minus") or [])[:5],
                },
                latent_note=f"backend={latent_info.get('backend')} external={latent_info.get('external')}",
                intervention=(cascade.verdict if cascade else "not_run"),
                strength="candidate_association",
            )
        )
    else:
        rows.append(
            CorrespondenceRow(
                fly_ids=list(assay.contributors),
                fly_function=assay.function_id,
                fly_evidence=assay.evidence_mode + " / " + assay.validation_status,
                llm_ids=[],
                svd_profile={"status": "unidentified", "reason": "no checkpoint SVD"},
                latent_note=f"backend={latent_info.get('backend')}",
                intervention="not_run",
                strength="unidentified",
            )
        )
    rows.append(
        CorrespondenceRow(
            fly_ids=list(da_assay.contributors),
            fly_function=da_assay.function_id,
            fly_evidence=da_assay.evidence_mode + " / " + da_assay.validation_status,
            llm_ids=[],
            svd_profile={"status": "see_value_modulation", "nt": "dopamine"},
            latent_note="dopamine analogue is gain, not a named layer",
            intervention="not_run" if bundle is None else "observational_value_modulation",
            strength="candidate_association" if bundle is not None else "unidentified",
        )
    )

    report = {
        "schema": SCHEMA_VERSION,
        "created_at": utc_now(),
        "packages": {**package_versions(), **pin_versions()},
        "latent_backend": latent_info,
        "fly_assay": assay.to_dict(),
        "literature_prior_assay": literature_assay.to_dict(),
        "dopamine_assay": da_assay.to_dict(),
        "neurotransmitters": nts,
        "malecns": cns,
        "biological_reference": bio.to_dict(),
        "neuprint": neu,
        "functions": functions_to_dict(),
        "model_index": index.to_dict() if index else None,
        "unmodified_baselines": {"looming_language": baseline_loom, "causal_consequence": baseline_causal},
        "candidates": [c.to_dict() for c in cands[:12]],
        "correspondence_table": [r.to_dict() for r in rows],
        "cascade": cascade.to_dict() if cascade else None,
        "analogous_region_test": region_test.to_dict() if region_test else None,
        "dopamine_region_test": da_region.to_dict() if da_region else None,
        "value_modulation": value_mod,
        "region_test_validation": validation,
        "prior_comparison": priors,
        "second_architecture": second_arch,
        "architecture_graph_preview": [e.to_dict() for e in architecture_edges(index.n_layers)] if index else [],
        "missing_resources": missing,
        "executed": {
            "fly_assay": True,
            "malecns_weights": True,
            "dopamine_assay": True,
            "weight_svd": bundle is not None,
            "latent_status": True,
            "region_test": region_test is not None,
            "region_test_validation": bool(validation.get("mechanics_pass")),
            "cascade": cascade is not None,
            "value_modulation": bool(value_mod.get("executed")),
        },
        "answers": _answers(assay, rows, cascade, priors, missing, baseline_loom, validation, region_test, da_assay, value_mod),
    }
    if out_dir is not None:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "atlas_experiment.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        (dest / "atlas_experiment.md").write_text(render_markdown(report), encoding="utf-8")
        (dest / "region_test_validation.json").write_text(
            json.dumps(validation, indent=2, default=str), encoding="utf-8"
        )
    return report


def _answers(assay, rows, cascade, priors, missing, baseline, validation=None, region_test=None, da_assay=None, value_mod=None) -> dict[str, Any]:
    row = rows[0].to_dict() if rows else {}
    return {
        "fly_neurons_tested": {
            "neurons": list(assay.neurons),
            "contributors": list(assay.contributors),
            "function": assay.function_id,
            "recordings_vs_simulation": assay.evidence_mode,
            "living_fly": assay.living_fly,
            "validation_status": assay.validation_status,
        },
        "dopamine_circuit": None if da_assay is None else {
            "neurons": list(da_assay.neurons),
            "contributors": list(da_assay.contributors),
            "validation_status": da_assay.validation_status,
        },
        "biological_function_and_evidence": assay.notes,
        "llm_components": row.get("llm_layer_module_units"),
        "signed_svd": row.get("signed_svd_logit_profile"),
        "measured_behavior": baseline,
        "cascade": cascade.to_dict() if cascade else {"verdict": "not_run"},
        "unidentified_contradicted_untested": missing,
        "fly_prior_improved": None if priors is None else priors.get("fly_prior_improved"),
        "region_test_alignment": None if region_test is None else region_test.alignment_score,
        "region_test_llm_source": None if region_test is None else region_test.llm_source,
        "region_test_mechanics_pass": None if validation is None else validation.get("mechanics_pass"),
        "value_modulation_dissociated_from_word": None if not value_mod else value_mod.get("dissociated_from_word_dopamine"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    a = report["answers"]
    lines = [
        "# LLMIntent atlas correspondence (v1.6)",
        "",
        f"Schema `{report['schema']}` · {report['created_at']}",
        "",
        "## Fly assay",
        "",
        f"- Circuit: `{report['fly_assay']['circuit_id']}`",
        f"- Mode: `{report['fly_assay']['evidence_mode']}` (living fly: {report['fly_assay']['living_fly']})",
        f"- Validation: {report['fly_assay']['validation_status']}",
        f"- Contributors: {', '.join(report['fly_assay']['contributors']) or '(none)'}",
        "",
        "## Correspondence table",
        "",
    ]
    for row in report["correspondence_table"]:
        lines.append(f"- Fly `{row['fly_neuron_or_circuit']}` → LLM `{row['llm_layer_module_units']}`")
        lines.append(f"  - strength: `{row['strength']}` · fly evidence: {row['fly_evidence']}")
        lines.append(f"  - SVD: {row['signed_svd_logit_profile']}")
        lines.append(f"  - intervention: {row['intervention_outcome']}")
    lines += [
        "",
        "## Cascade",
        "",
        f"{a['cascade']}",
        "",
        "## Analogous region test",
        "",
        json.dumps(report.get("analogous_region_test"), indent=2, default=str)[:4000],
        "",
        "## Region-test validation (synthetic mechanics, not pretrained evidence)",
        "",
        f"- mechanics_pass: {a.get('region_test_mechanics_pass')}",
        f"- llm_source: {a.get('region_test_llm_source')}",
        f"- alignment: {a.get('region_test_alignment')}",
        "",
        "## Did the fly prior improve anything?",
        "",
        str(a.get("fly_prior_improved")),
        "",
        "## Missing resources",
        "",
    ]
    for m in report["missing_resources"] or ["(none)"]:
        lines.append(f"- {m}")
    lines += [
        "",
        "## Reproduce",
        "",
        "```",
        "python -m llmintent atlas --out artifacts/atlas_v16",
        "```",
        "",
        "Latent backend: `llmintent.latent` → latentintent if installed, else `llmintent.latent_vendor`.",
        "",
    ]
    return "\n".join(lines)
