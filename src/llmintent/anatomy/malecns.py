"""Versioned MaleCNS reference from the local fly-brain download.

The literature-core graph remains a labeled prior. Synapse counts and
consensus neurotransmitters come from MaleCNS v1.0 traced tables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "malecns_subset.json"


@dataclass
class BioNeuron:
    type_id: str
    region: str
    fly_atlas_region: str
    citations: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    supported_function: str
    proposed_function: str
    unresolved: str
    nt_prediction: str | None = None
    nt_confidence: str | None = None
    evidence_mode: str = "literature_prior"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "region": self.region,
            "fly_atlas_region": self.fly_atlas_region,
            "citations": list(self.citations),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "supported_function": self.supported_function,
            "proposed_function": self.proposed_function,
            "unresolved": self.unresolved,
            "nt_prediction": self.nt_prediction,
            "nt_confidence": self.nt_confidence,
            "evidence_mode": self.evidence_mode,
        }


@dataclass
class BioReference:
    dataset: str
    version_retrieved: str
    retrieval: str
    specimen: str
    edge_weight_meaning: str
    neurons: list[BioNeuron] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    literature_prior_separate: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version_retrieved": self.version_retrieved,
            "retrieval": self.retrieval,
            "specimen": self.specimen,
            "edge_weight_meaning": self.edge_weight_meaning,
            "neurons": [n.to_dict() for n in self.neurons],
            "edges": list(self.edges),
            "literature_prior_separate": self.literature_prior_separate,
            "notes": list(self.notes),
        }


_BUNDLED: tuple[BioNeuron, ...] = (
    BioNeuron(
        "LPLC2", "lobula plate / lobula", "vision",
        ("Klapoetke et al. 2017 Nature 543:96-100",),
        ("T4", "T5"), ("DNp01", "DNp10"),
        "Looming-object feature discrimination (expanding dark objects).",
        "May contribute to other object expansion statistics.",
        "Exact MaleCNS synapse counts not retrieved in this workspace.",
        nt_prediction="ACh (prediction)", nt_confidence="literature_not_measured_here",
    ),
    BioNeuron(
        "LC4", "lobula", "vision",
        ("Klapoetke et al. 2017 Nature 543:96-100",),
        ("T4", "T5"), ("DNp01",),
        "Looming / object-approach visual projection.",
        "Partial overlap with LPLC2; not interchangeable.",
        "Relative weight vs LPLC2 on MaleCNS not retrieved.",
    ),
    BioNeuron(
        "DNp01", "descending / giant fibre", "descending",
        ("von Reyn et al. 2014 Nat Neurosci 17:962-970",),
        ("LPLC2", "LC4"), ("GFC",),
        "Short-latency escape jump when activated.",
        "Action selection / command.",
        "Living-fly silencing of LPLC2 vs DNp01 not re-run here.",
    ),
    BioNeuron(
        "KC", "mushroom body", "associative",
        ("Aso et al. 2014 eLife 3:e04577",),
        ("ALPN",), ("MBON",),
        "Sparse associative binding of cues to outcomes.",
        "Episodic-like association analogue.",
        "Not the circuit of the first assay.",
    ),
)


def bundled_reference() -> BioReference:
    from llmintent.anatomy.flycns import load_circuit_bundle, majority_nt, malecns_status

    status = malecns_status()
    circuit = load_circuit_bundle()
    neurons = []
    for name, row in (circuit.get("populations") or {}).items():
        nt = majority_nt(row.get("nt") or {})
        neurons.append(
            BioNeuron(
                type_id=name,
                region=name,
                fly_atlas_region="associative" if name in {"PAM", "PPL1", "KC", "MBON", "APL"} else "vision",
                citations=("MaleCNS v1.0 traced + consensus_nt",),
                inputs=(),
                outputs=(),
                supported_function=f"Majority NT {nt} on {row.get('n')} cells.",
                proposed_function="See fly assays for tested function.",
                unresolved="Stimulus tuning is not in the EM volume.",
                nt_prediction=nt,
                nt_confidence="consensus_nt majority on extracted population",
                evidence_mode="measured",
            )
        )
    return BioReference(
        dataset="MaleCNS v1.0",
        version_retrieved="traced-only minconf-0.5 (local fly-brain download)",
        retrieval=str(circuit.get("retrieval")),
        specimen="adult male Drosophila, one fixed brain+VNC",
        edge_weight_meaning=str(circuit.get("weight_meaning")),
        neurons=neurons or list(_BUNDLED),
        edges=list(circuit.get("edges") or []),
        notes=[
            "Literature-core graph remains a separate prior (anatomy.connectome).",
            "Live fly-brain root: " + str(status.get("flybrain_root")),
            str(circuit.get("monoamine_note")),
        ],
    )


def load_local_export(path: str | Path) -> BioReference:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    neurons = [
        BioNeuron(
            type_id=n["type_id"],
            region=n.get("region", ""),
            fly_atlas_region=n.get("fly_atlas_region", ""),
            citations=tuple(n.get("citations") or ()),
            inputs=tuple(n.get("inputs") or ()),
            outputs=tuple(n.get("outputs") or ()),
            supported_function=n.get("supported_function", ""),
            proposed_function=n.get("proposed_function", ""),
            unresolved=n.get("unresolved", ""),
            nt_prediction=n.get("nt_prediction"),
            nt_confidence=n.get("nt_confidence"),
            evidence_mode=n.get("evidence_mode", "local_export"),
        )
        for n in payload.get("neurons") or []
    ]
    return BioReference(
        dataset=payload.get("dataset", "local"),
        version_retrieved=payload.get("version_retrieved", "local"),
        retrieval="local_export",
        specimen=payload.get("specimen", "local"),
        edge_weight_meaning=payload.get("edge_weight_meaning", "see file"),
        neurons=neurons,
        edges=list(payload.get("edges") or []),
        notes=list(payload.get("notes") or []),
    )


def neuprint_adapter() -> dict[str, Any]:
    server = os.environ.get("NEUPRINT_SERVER")
    dataset = os.environ.get("NEUPRINT_DATASET", "male-cns:latest")
    if not server:
        return {
            "available": False,
            "capability": "unavailable",
            "reason": "NEUPRINT_SERVER unset; MaleCNS not retrieved",
            "dataset": dataset,
        }
    return {
        "available": True,
        "capability": "configured_not_fetched_in_this_run",
        "server": server,
        "dataset": dataset,
        "note": "Adapter is present; this run does not pull the full connectome.",
    }


def write_bundled_json(path: Path | None = None) -> Path:
    dest = path or DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(bundled_reference().to_dict(), indent=2), encoding="utf-8")
    return dest
