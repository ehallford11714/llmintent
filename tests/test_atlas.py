"""Atlas v1.6: fly assay, SVD convention, unidentified behavior, region test."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn


def test_malecns_circuits_use_downloaded_nts():
    from llmintent.anatomy.flycns import load_circuit_bundle, majority_nt
    from llmintent.anatomy.fly_assay import run_dopamine_malecns_assay, run_looming_malecns_assay

    c = load_circuit_bundle()
    assert c["populations"]["LPLC2"]["nt"]["acetylcholine"] == 185
    assert majority_nt(c["populations"]["PAM"]["nt"]) == "dopamine"
    assert majority_nt(c["populations"]["PPL1"]["nt"]) == "dopamine"
    assert majority_nt(c["populations"]["APL"]["nt"]) == "gaba"
    loom = run_looming_malecns_assay()
    assert loom.living_fly is False
    assert "malecns" in loom.circuit_id
    held = [r for r in loom.results if r.held_out and r.perturbation == "none"]
    expand = np.mean([r.readout for r in held if r.stimulus.startswith("expanding")])
    recede = np.mean([r.readout for r in held if r.stimulus.startswith("receding")])
    assert expand > recede
    da = run_dopamine_malecns_assay()
    assert "dopamine" in da.validation_status or da.contributors
    pam = next(r for r in da.results if r.stimulus == "cue_PAM_held")
    ppl = next(r for r in da.results if r.stimulus == "cue_PPL1_held")
    cue = next(r for r in da.results if r.stimulus == "cue_only_held")
    assert pam.readout > cue.readout > ppl.readout


def test_transmitter_analogues_are_jobs_not_layer_names():
    from llmintent.anatomy.transmitters import transmitter_report

    rep = transmitter_report()
    da = next(a for a in rep["analogues"] if a["fly_nt"] == "dopamine")
    assert "gain" in da["llm_operation"].lower()
    assert "not" in da["not_a_claim"].lower()


def test_fly_assay_reproduces_held_out_direction():
    from llmintent.anatomy.fly_assay import assay_signature, run_looming_giant_fibre_assay

    assay = run_looming_giant_fibre_assay()
    assert assay.living_fly is False
    assert assay.evidence_mode == "proposed_simulation"
    assert "LPLC2" in assay.contributors
    assert assay.validation_status.startswith("simulation_reproduces")
    held = [r for r in assay.results if r.held_out and r.perturbation == "none"]
    expand = np.mean([r.readout for r in held if r.stimulus.startswith("expanding")])
    recede = np.mean([r.readout for r in held if r.stimulus.startswith("receding")])
    assert expand > recede
    silenced = next(r for r in assay.results if r.perturbation == "silence_LPLC2")
    rescued = next(r for r in assay.results if r.perturbation == "rescue_LPLC2")
    control = next(r for r in assay.results if r.perturbation == "silence_T4_control")
    intact = next(
        r for r in assay.results if r.stimulus == silenced.stimulus and r.perturbation == "none" and r.held_out
    )
    assert intact.readout - silenced.readout > intact.readout - control.readout
    assert rescued.readout > silenced.readout
    sig = assay_signature(assay)
    assert "Approaching-object" in sig["shared_operation"]
    d = assay.to_dict()
    json.dumps(d)
    assert d["fit_conditions"] != d["eval_conditions"]


def test_ffn_down_svd_convention_and_sign_invariance():
    from llmintent.anatomy.spaces import (
        coefficient,
        down_weight_d_by_m,
        reconstruction_error,
        sign_invariant_subspace_overlap,
        svd_down,
    )

    d, m = 8, 16
    W = np.zeros((d, m))
    W[:, 0] = np.eye(d)[0]
    W[:, 1] = np.eye(d)[1] * 0.5
    lin = nn.Linear(m, d, bias=False)
    lin.weight.data = torch.tensor(W, dtype=torch.float32)
    Wdm = down_weight_d_by_m(lin.weight, "linear_out_in")
    assert Wdm.shape == (d, m)
    U, S, V = svd_down(Wdm, top_k=2)
    err = reconstruction_error(Wdm, U, S, V)
    assert err < 1e-6
    # Leading write direction aligns with e0 (sign-invariant).
    assert abs(float(U[0, 0])) > 0.9
    z = np.zeros(m)
    z[0] = 2.0
    a0 = coefficient(float(S[0]), V[:, 0], z)
    contrib = a0 * U[:, 0]
    assert abs(contrib[0]) > abs(contrib[1])
    overlap = sign_invariant_subspace_overlap(U, -U)
    assert overlap > 0.99
    # Conv1D layout: [m, d]
    conv = torch.tensor(W.T, dtype=torch.float32)
    W2 = down_weight_d_by_m(conv, "conv1d_in_out")
    assert np.allclose(W2, Wdm)


def test_match_text_abstains_below_floor():
    from llmintent.anatomy.evidence import UNIDENTIFIED
    from llmintent.anatomy.svd_map import match_text_to_region, region_axis_from_anatomy
    from llmintent.anatomy.svd_map import SVDAnatomy

    rid, score = match_text_to_region("zzzzqxqxqx not a region")
    assert rid == UNIDENTIFIED
    empty = SVDAnatomy(kind="weights", occupancy={})
    axis, src = region_axis_from_anatomy(empty, "vision", 4)
    assert axis is None
    assert src == UNIDENTIFIED
    axis2, src2 = region_axis_from_anatomy(empty, "vision", 4, allow_random_control=True, control_seed=1)
    assert axis2 is not None and src2 == "random_control"


def test_benign_car_stop_is_experimental_not_misalign():
    from llmintent.anatomy.misalign import scan_negative_intent

    _loci, flag = scan_negative_intent("I see a car. Stop.")
    assert flag.triggered is False
    shorts = [x for x in _loci if x.kind == "short_pipe"]
    if shorts:
        assert all(x.experimental_signal for x in shorts)
    _loci2, flag2 = scan_negative_intent("I see a car because the light is red. Stop.")
    assert flag2.triggered is False


def test_noop_hooks_leave_weights():
    from llmintent.anatomy.cascade import no_op_hooks
    from llmintent.models import ModelBundle

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            layer = nn.Linear(4, 4)
            self.transformer.h = nn.ModuleList([layer])
            self.weight = layer.weight

        def forward(self, x):
            return self.transformer.h[0](x)

    model = Tiny()
    before = model.weight.detach().clone()
    bundle = ModelBundle("tiny", tokenizer=None, model=model, device=torch.device("cpu"), is_causal=True, num_layers=1)
    with no_op_hooks(bundle):
        model.weight.sum().backward() if False else None
        _ = model(torch.ones(1, 4))
    assert torch.allclose(before, model.weight.detach())


def test_analogous_region_test_without_checkpoint():
    from llmintent.anatomy.fly_assay import run_looming_giant_fibre_assay
    from llmintent.anatomy.region_test import test_analogous_region
    from llmintent.anatomy.tasks import by_split

    assay = run_looming_giant_fibre_assay()
    result = test_analogous_region(assay, None, by_split("looming_language", "confirmation"))
    assert result.llm_source == "unavailable"
    assert result.fly_source == "proposed_simulation"
    assert "cosine" not in " ".join(result.notes).lower() or "No cosine" in " ".join(result.notes)
    assert result.fly_vector["approach"] is not None
    assert result.fly_vector["approach"] > result.fly_vector["recede"]


def test_region_test_validation_mechanics():
    from llmintent.anatomy.region_test import validate_analogous_region_test

    report = validate_analogous_region_test()
    assert report["source"] == "synthetic"
    assert report["mechanics_pass"] is True
    assert report["matched"]["alignment"] > report["anti"]["alignment"]
    assert report["matched"]["alignment"] > (report["shuffled"]["alignment"] or -1)
    assert "pretrained" not in report["note"].lower() or "not evidence" in report["note"].lower()


def test_experiment_no_model_writes_report(tmp_path):
    from llmintent.anatomy.experiment import run_atlas_experiment

    report = run_atlas_experiment(model=None, out_dir=tmp_path)
    assert report["executed"]["fly_assay"] is True
    assert report["executed"]["weight_svd"] is False
    assert report["executed"]["region_test"] is True
    assert report["analogous_region_test"]["llm_source"] == "unavailable"
    assert report["region_test_validation"]["mechanics_pass"] is True
    assert report["correspondence_table"][0]["strength"] in {"unidentified", "candidate_association"}
    assert (tmp_path / "atlas_experiment.json").exists()
    assert (tmp_path / "atlas_experiment.md").exists()
    assert (tmp_path / "region_test_validation.json").exists()
    assert "lplc2" in report["fly_assay"]["circuit_id"]


def test_latent_backend_contract():
    from llmintent import latent
    from llmintent.anatomy.capture import latent_backend_status

    info = latent_backend_status()
    assert info["available"] is True
    assert latent.backend_name() in {"latentintent", "latentintentinspect", "llmintent.latent_vendor"}
    assert "latent_vendor" in (info.get("backend") or "") or info.get("external") is True


def test_cli_atlas_no_model(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [
            sys.executable, "-m", "llmintent", "atlas", "--no-model",
            "--out", str(tmp_path), "--format", "json",
        ],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else json.loads((tmp_path / "atlas_experiment.json").read_text(encoding="utf-8"))
    assert payload["fly_assay"]["living_fly"] is False
