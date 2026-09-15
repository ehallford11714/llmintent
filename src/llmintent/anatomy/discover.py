"""Primary discovery: weight SVD → signed logit contrasts → FFN unit localization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llmintent.anatomy.adapters import ModelIndex, index_bundle
from llmintent.anatomy.evidence import UNIDENTIFIED, ClaimConfidences, EvidenceClaim
from llmintent.anatomy.spaces import decompose_ffn_down, signed_logit_profile
from llmintent.anatomy.tasks import TaskItem


@dataclass
class LocalizedUnit:
    module_path: str
    layer: int
    unit: int
    loading: float
    signed_contribution: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": f"{self.module_path}:unit{self.unit}",
            "layer": self.layer,
            "unit": self.unit,
            "loading": round(self.loading, 4),
            "signed_contribution": self.signed_contribution,
        }


@dataclass
class CandidateRegion:
    region_id: str
    function_id: str
    layer: int
    component_index: int
    sign: int
    module_path: str
    contrast: float
    singular_value: float
    units: list[LocalizedUnit]
    logit_profile: dict[str, Any]
    source: str = "inferred"
    strength: str = "candidate_association"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "function_id": self.function_id,
            "layer": self.layer,
            "component_index": self.component_index,
            "sign": self.sign,
            "module_path": self.module_path,
            "contrast": round(self.contrast, 4),
            "singular_value": round(self.singular_value, 4),
            "units": [u.to_dict() for u in self.units],
            "logit_profile": self.logit_profile,
            "source": self.source,
            "strength": self.strength,
            "notes": list(self.notes),
            "claim": EvidenceClaim(
                statement=(
                    f"Layer {self.layer} SVD component {self.component_index} "
                    f"(sign {self.sign:+d}) is a candidate for {self.function_id}"
                ),
                source=self.source,
                strength=self.strength,
                confidences=ClaimConfidences(
                    observational=min(1.0, abs(self.contrast) / 10.0),
                    intervention=None,
                    biological_analogy=None,
                    missing=("intervention", "held_out") if self.strength == "candidate_association" else (),
                ),
            ).to_dict(),
        }


def _token_ids(tokenizer: Any, pieces: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for p in pieces:
        got = tokenizer(p, add_special_tokens=False)["input_ids"]
        ids.extend(int(x) for x in got)
    return ids


def _ffn_down_specs(index: ModelIndex) -> list:
    return [c for c in index.components if c.role == "ffn_out" and c.capability == "ok"]


def _get_module(model: Any, path: str) -> Any:
    obj = model
    # strip trailing .weight
    parts = path.replace(".weight", "").split(".")
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def discover_from_weights(
    bundle: Any,
    items: list[TaskItem],
    *,
    function_id: str,
    top_k: int = 6,
    max_layers: int | None = None,
    layer_filter: list[int] | None = None,
    prior_layers: list[int] | None = None,
) -> list[CandidateRegion]:
    """Rank SVD components by task logit contrast, not by largest singular value alone."""
    index = index_bundle(bundle)
    tok = bundle.tokenizer
    pos_ids = _token_ids(tok, items[0].positive)
    neg_ids = _token_ids(tok, items[0].negative)
    from llmintent.models import get_unembedding_matrix

    try:
        unembed = get_unembedding_matrix(bundle.model)
    except Exception:
        return []

    specs = _ffn_down_specs(index)
    if max_layers is not None:
        specs = [s for s in specs if s.component.layer is not None and s.component.layer < max_layers]
    if layer_filter is not None:
        allow = set(layer_filter)
        specs = [s for s in specs if s.component.layer in allow]

    ranked: list[CandidateRegion] = []
    for spec in specs:
        layer = int(spec.component.layer or 0)
        try:
            mod = _get_module(bundle.model, spec.component.module_path)
            weight = mod.data if hasattr(mod, "data") else mod.weight.data
        except Exception:
            continue
        comps = decompose_ffn_down(
            weight,
            layout=spec.layout,
            layer=layer,
            module_path=spec.component.module_path,
            top_k=top_k,
        )
        for comp in comps:
            for sign in (1, -1):
                profile = signed_logit_profile(
                    sign * comp.u,
                    unembed,
                    tok,
                    contrast_ids={"task_pos": pos_ids, "task_neg": neg_ids},
                )
                contrasts = profile.get("contrasts") or {}
                plus = float((contrasts.get("task_pos") or {}).get("plus", 0.0))
                minus = float((contrasts.get("task_neg") or {}).get("plus", 0.0))
                contrast = plus - minus
                units = [
                    LocalizedUnit(spec.component.module_path, layer, u, load * sign)
                    for u, load in comp.unit_loadings[:8]
                ]
                ranked.append(
                    CandidateRegion(
                        region_id=f"svd:L{layer}:k{comp.index}:s{sign:+d}",
                        function_id=function_id,
                        layer=layer,
                        component_index=comp.index,
                        sign=sign,
                        module_path=spec.component.module_path,
                        contrast=contrast,
                        singular_value=comp.singular_value,
                        units=units,
                        logit_profile=profile,
                        notes=[
                            "Primary pipeline: SVD of W_down → signed W_vocab u contrast.",
                            "Lower singular values retained within top_k.",
                            "Depth-band prior not applied unless prior_layers was passed.",
                        ],
                    )
                )
    ranked.sort(key=lambda c: -abs(c.contrast))
    if prior_layers is not None:
        prefer = [c for c in ranked if c.layer in set(prior_layers)]
        rest = [c for c in ranked if c.layer not in set(prior_layers)]
        # Same budget: keep len(ranked) but prior only reorders a comparable prefix.
        ranked = prefer + rest
    return ranked


def shuffled_labels(cands: list[CandidateRegion], seed: int = 0) -> list[CandidateRegion]:
    rng = np.random.default_rng(seed)
    labels = [c.function_id for c in cands]
    rng.shuffle(labels)
    out = []
    for c, lab in zip(cands, labels):
        out.append(
            CandidateRegion(
                **{**c.__dict__, "function_id": lab, "notes": list(c.notes) + ["shuffled_function_label"]},
            )
        )
    return out
