"""Load MaleCNS circuits from the local fly-brain download.

Weights and neurotransmitter calls come from the traced-only Feather tables
already on disk under ``fly-brain/data``. Stimulus selectivity is still a
literature prior: the connectome does not contain recordings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CIRCUIT_PATH = Path(__file__).parent / "data" / "malecns_circuits.json"

_FLYBRAIN_CANDIDATES = (
    Path(os.environ["FLYBRAIN_ROOT"]) if os.environ.get("FLYBRAIN_ROOT") else None,
    Path(__file__).resolve().parents[4] / "fly-brain",
    Path.home() / "Desktop" / "research" / "fly-brain",
    Path(r"c:\Users\ehall\Desktop\research\fly-brain"),
)


def flybrain_root() -> Path | None:
    for cand in _FLYBRAIN_CANDIDATES:
        if cand is None:
            continue
        data = cand / "data"
        if (data / "body-neurotransmitters-male-cns-v1.0.feather").exists() and (
            data / "_cache-connectome-traced.npz"
        ).exists():
            return cand
    return None


def load_circuit_bundle(path: Path | None = None) -> dict[str, Any]:
    dest = path or CIRCUIT_PATH
    return json.loads(dest.read_text(encoding="utf-8"))


def majority_nt(counts: dict[str, int]) -> str:
    if not counts:
        return "unclear"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def edge_map(bundle: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(e["pre"], e["post"]): e for e in bundle.get("edges") or []}


def malecns_status() -> dict[str, Any]:
    root = flybrain_root()
    bundle = load_circuit_bundle()
    return {
        "available": True,
        "live_flybrain": root is not None,
        "flybrain_root": str(root) if root else None,
        "dataset": bundle.get("dataset"),
        "edge_set": bundle.get("edge_set"),
        "retrieval": bundle.get("retrieval"),
        "n_neurons_full": bundle.get("n_neurons_full"),
        "n_edges_full": bundle.get("n_edges_full"),
        "note": (
            "Circuit JSON was extracted from the downloaded MaleCNS traced graph. "
            "Stimulus tuning is not in the EM volume."
        ),
    }
