"""Activation capture with reuse contracts. Interventions never reuse a clean run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llmintent.anatomy.evidence import (
    CaptureContract,
    ExperimentRecord,
    hash_prompt,
    package_versions,
    utc_now,
)


@dataclass
class CaptureStore:
    records: dict[str, ExperimentRecord] = field(default_factory=dict)

    def get(self, contract: CaptureContract, prompt: str) -> ExperimentRecord | None:
        rec = self.records.get(contract.key() + hash_prompt(prompt))
        if rec is None or rec.contract is None:
            return None
        if rec.contract.compatible_with(contract) and rec.prompt_hash == hash_prompt(prompt):
            if contract.intervention_id is not None:
                return None
            return rec
        return None

    def put(self, rec: ExperimentRecord) -> ExperimentRecord:
        key = (rec.contract.key() if rec.contract else rec.run_id) + rec.prompt_hash
        self.records[key] = rec
        return rec


def latent_backend_status() -> dict[str, Any]:
    from llmintent import latent

    info = latent.describe()
    info["adapter"] = (
        "llmintent.latent prefers latentintent / latentintentinspect, else "
        "llmintent.latent_vendor. No invented extra package."
    )
    return info


def capture_hidden_last_token(
    bundle: Any,
    prompt: str,
    *,
    store: CaptureStore | None = None,
    intervention_id: str | None = None,
    sites: tuple[str, ...] = ("residual_last_token",),
) -> ExperimentRecord:
    from llmintent.forward import forward_hidden_states

    cfg = getattr(bundle.model, "config", None)
    contract = CaptureContract(
        model_id=str(bundle.name),
        revision=getattr(cfg, "_name_or_path", None),
        sites=sites,
        intervention_id=intervention_id,
        dtype=str(getattr(cfg, "torch_dtype", "float32")),
    )
    store = store or CaptureStore()
    if intervention_id is None:
        cached = store.get(contract, prompt)
        if cached is not None:
            cached.notes = list(cached.notes) + ["reused_compatible_capture"]
            return cached
    _, states = forward_hidden_states(bundle, prompt)
    rows = []
    for i, state in enumerate(states):
        h = state[0, -1, :].detach().float().cpu().numpy()
        rows.append(h.tolist())
    rec = ExperimentRecord(
        run_id=contract.key() + hash_prompt(prompt) + (intervention_id or "clean"),
        prompt=prompt,
        prompt_hash=hash_prompt(prompt),
        contract=contract,
        source="measured",
        activations={
            "site": "residual_last_token",
            "n_states": len(states),
            "embedding_index": 0,
            "note": "states[0] is embeddings; block i is states[i+1]. Not interchangeable without this map.",
            "last_token": rows,
        },
        packages=package_versions(),
        notes=["Clean capture" if intervention_id is None else f"Intervention {intervention_id}"],
        executed=True,
    )
    if intervention_id is None:
        store.put(rec)
    return rec


def last_token_matrix(rec: ExperimentRecord) -> np.ndarray:
    data = rec.activations.get("last_token") or []
    return np.asarray(data, dtype=np.float64)
