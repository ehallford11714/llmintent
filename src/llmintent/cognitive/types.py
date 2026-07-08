"""Shared types for cognitive module kernels."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch

COGNITIVE_MODULES = ("identity", "reasoning", "meta_reasoning", "ideation")


@dataclass
class CognitiveKernel:
    module: str
    layer: int
    score: float
    kl_weight: float
    barlow_invariance: float
    barlow_redundancy: float
    kernel_basis: torch.Tensor
    top_intent: str = ""


@dataclass
class CognitiveModuleProfile:
    twin_a: str
    twin_b: str
    kl_profile: list[float]
    barlow_metrics: dict[str, float]
    kernels: list[CognitiveKernel] = field(default_factory=list)
    layer_assignments: pd.DataFrame = field(default_factory=pd.DataFrame)
    projector: torch.Tensor | None = None

    def module_layers(self, module: str) -> list[int]:
        if self.layer_assignments.empty:
            return []
        mask = self.layer_assignments["dominant_module"] == module
        return self.layer_assignments.loc[mask, "layer"].astype(int).tolist()

    def to_dict(self) -> dict:
        return {
            "twin_a": self.twin_a,
            "twin_b": self.twin_b,
            "kl_profile": self.kl_profile,
            "barlow_metrics": self.barlow_metrics,
            "module_layers": {m: self.module_layers(m) for m in COGNITIVE_MODULES},
            "kernels": [
                {
                    "module": k.module,
                    "layer": k.layer,
                    "score": k.score,
                    "kl_weight": k.kl_weight,
                    "barlow_invariance": k.barlow_invariance,
                    "barlow_redundancy": k.barlow_redundancy,
                    "top_intent": k.top_intent,
                }
                for k in self.kernels
            ],
        }
