"""Shared evidence contract for structural and runtime anatomy (v1.6).

Every claim carries a source kind. Measured, inferred, prior, synthetic,
unavailable, and unidentified results are not interchangeable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "llmintent.anatomy.evidence/v1"
ACCEPTANCE_FLOOR = 0.18
UNIDENTIFIED = "unidentified"

SOURCE_KINDS = (
    "measured",
    "inferred",
    "prior",
    "synthetic",
    "unavailable",
    "unidentified",
    "random_control",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_prompt(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def hash_payload(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class ComponentId:
    """Stable identifier for one model tensor (or a slice of it)."""

    checkpoint: str
    revision: str | None
    module_path: str
    layer: int | None = None
    head: int | None = None
    expert: int | None = None
    feature: int | None = None

    @property
    def id(self) -> str:
        parts = [self.checkpoint, self.revision or "unknown", self.module_path]
        if self.layer is not None:
            parts.append(f"L{self.layer}")
        if self.head is not None:
            parts.append(f"H{self.head}")
        if self.expert is not None:
            parts.append(f"E{self.expert}")
        if self.feature is not None:
            parts.append(f"F{self.feature}")
        return "|".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "module_path": self.module_path,
            "layer": self.layer,
            "head": self.head,
            "expert": self.expert,
            "feature": self.feature,
        }


@dataclass
class ComponentSpec:
    component: ComponentId
    shape: tuple[int, ...]
    orientation: str
    in_space: str
    out_space: str
    dtype: str
    quantization: str | None = None
    layout: str = "linear_out_in"
    nonlinearity: str | None = None
    role: str = "unknown"
    notes: list[str] = field(default_factory=list)
    capability: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.to_dict(),
            "shape": list(self.shape),
            "orientation": self.orientation,
            "in_space": self.in_space,
            "out_space": self.out_space,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "layout": self.layout,
            "nonlinearity": self.nonlinearity,
            "role": self.role,
            "capability": self.capability,
            "notes": list(self.notes),
        }


@dataclass
class ClaimConfidences:
    """Keep analogy, observation, and intervention separate. Never collapse."""

    biological_analogy: float | None = None
    observational: float | None = None
    intervention: float | None = None
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "biological_analogy": self.biological_analogy,
            "observational": self.observational,
            "intervention": self.intervention,
            "missing": list(self.missing),
        }


@dataclass
class EvidenceClaim:
    statement: str
    source: str
    strength: str = "candidate_association"
    confidences: ClaimConfidences = field(default_factory=ClaimConfidences)
    alternatives: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "source": self.source,
            "strength": self.strength,
            "confidences": self.confidences.to_dict(),
            "alternatives": list(self.alternatives),
            "provenance": list(self.provenance),
        }


@dataclass
class CaptureContract:
    model_id: str
    revision: str | None
    sites: tuple[str, ...]
    generation_step: int = 0
    thinking: bool = False
    dtype: str = "float32"
    intervention_id: str | None = None
    chat_template: str = "default"

    def key(self) -> str:
        return hash_payload(
            {
                "model": self.model_id,
                "revision": self.revision,
                "sites": list(self.sites),
                "step": self.generation_step,
                "thinking": self.thinking,
                "dtype": self.dtype,
                "intervention": self.intervention_id,
                "chat": self.chat_template,
            }
        )

    def compatible_with(self, other: "CaptureContract") -> bool:
        return self.key() == other.key()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "sites": list(self.sites),
            "generation_step": self.generation_step,
            "thinking": self.thinking,
            "dtype": self.dtype,
            "intervention_id": self.intervention_id,
            "chat_template": self.chat_template,
            "key": self.key(),
        }


@dataclass
class ExperimentRecord:
    """One stored run: either clean capture or a specified intervention."""

    run_id: str
    schema: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    prompt: str = ""
    prompt_hash: str = ""
    task_id: str | None = None
    split: str | None = None
    contract: CaptureContract | None = None
    source: str = "measured"
    components: list[str] = field(default_factory=list)
    region_ids: list[str] = field(default_factory=list)
    activations: dict[str, Any] = field(default_factory=dict)
    intervention: dict[str, Any] | None = None
    outcome: dict[str, Any] = field(default_factory=dict)
    claims: list[EvidenceClaim] = field(default_factory=list)
    packages: dict[str, str] = field(default_factory=dict)
    seed: int | None = None
    notes: list[str] = field(default_factory=list)
    executed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema": self.schema,
            "created_at": self.created_at,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash or hash_prompt(self.prompt),
            "task_id": self.task_id,
            "split": self.split,
            "contract": self.contract.to_dict() if self.contract else None,
            "source": self.source,
            "components": list(self.components),
            "region_ids": list(self.region_ids),
            "activations": self.activations,
            "intervention": self.intervention,
            "outcome": self.outcome,
            "claims": [c.to_dict() for c in self.claims],
            "packages": dict(self.packages),
            "seed": self.seed,
            "notes": list(self.notes),
            "executed": self.executed,
        }


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("llmintent", "torch", "transformers", "numpy"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "unavailable"
    return out


def require_source(kind: str) -> str:
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown evidence source {kind!r}")
    return kind


def unidentified_claim(reason: str, *, provenance: Iterable[str] = ()) -> EvidenceClaim:
    return EvidenceClaim(
        statement=reason,
        source="unidentified",
        strength="unidentified",
        confidences=ClaimConfidences(missing=("axis",)),
        provenance=list(provenance),
    )


def merge_notes(*groups: Iterable[str] | None) -> list[str]:
    seen: list[str] = []
    for group in groups:
        if not group:
            continue
        for item in group:
            if item and item not in seen:
                seen.append(item)
    return seen


def as_float_map(values: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in values.items()}
