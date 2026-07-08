"""Orchestrate cognitive module kernel identification via KL + twin Barlow."""

from __future__ import annotations

import pandas as pd
import torch

from llmintent.cognitive.identity import extract_identity_kernel, identity_layer_scores
from llmintent.cognitive.ideation import extract_ideation_kernel, ideation_layer_scores
from llmintent.cognitive.meta_reasoning import extract_meta_reasoning_kernel, meta_reasoning_layer_scores
from llmintent.cognitive.reasoning import extract_reasoning_kernel, reasoning_layer_scores
from llmintent.cognitive.types import COGNITIVE_MODULES, CognitiveKernel, CognitiveModuleProfile
from llmintent.jspace.trace import build_intent_trace
from llmintent.kernels.barlow import extract_kernel_basis, minimize_twin_barlow, per_layer_barlow_features
from llmintent.kernels.kl_kernel import (
    collect_twin_hidden_matrix,
    kl_weighted_difference_kernel,
    per_layer_kl_profile,
)
from llmintent.models import ModelBundle
from llmintent.jspace.transport import TransportMaps

_MODULE_EXTRACTORS = {
    "identity": extract_identity_kernel,
    "reasoning": extract_reasoning_kernel,
    "meta_reasoning": extract_meta_reasoning_kernel,
    "ideation": extract_ideation_kernel,
}


def build_cognitive_module_profile(
    bundle: ModelBundle,
    twin_a: str,
    twin_b: str,
    *,
    transport: TransportMaps | None = None,
    position: int = -1,
    proj_dim: int = 32,
    kernel_rank: int = 4,
    cot_delta: torch.Tensor | None = None,
) -> CognitiveModuleProfile:
    """
    Identify identity, reasoning, meta-reasoning, and ideation kernels.

    Pipeline:
    1. Per-layer KL profile between twin prompts
    2. Twin Barlow minimization on hidden-state trajectories
    3. KL-weighted SVD kernel basis
    4. Layer scoring → dominant cognitive module assignment
    """
    kl_profile, _ = per_layer_kl_profile(bundle, twin_a, twin_b, position=position)
    h_a, h_b = collect_twin_hidden_matrix(bundle, twin_a, twin_b, position=position)

    projector, barlow_metrics = minimize_twin_barlow(
        h_a, h_b, kl_profile, proj_dim=proj_dim
    )
    barlow_basis = extract_kernel_basis(projector)
    kl_basis = kl_weighted_difference_kernel(h_a, h_b, kl_profile, top_k=kernel_rank)
    combined_basis = torch.cat([barlow_basis[:kernel_rank], kl_basis], dim=0)
    _, _, vh = torch.linalg.svd(combined_basis, full_matrices=False)
    master_basis = vh[:kernel_rank, :]

    barlow_diag, barlow_off = per_layer_barlow_features(h_a, h_b)

    trace_a = build_intent_trace(bundle, twin_a, transport=transport, position=position)
    entropy = torch.tensor(trace_a.entropy, dtype=torch.float32)
    occupancy = torch.tensor(trace_a.occupancy, dtype=torch.float32)
    motor = torch.tensor(trace_a.layer_stats["motor_alignment"].tolist(), dtype=torch.float32)

    id_scores = identity_layer_scores(kl_profile, barlow_diag, motor_alignment=motor)
    re_scores = reasoning_layer_scores(kl_profile, occupancy, entropy=entropy)
    meta_scores = meta_reasoning_layer_scores(kl_profile, barlow_off, cot_delta=cot_delta)
    ide_scores = ideation_layer_scores(entropy, motor, kl_profile=kl_profile)

    score_matrix = torch.stack([id_scores, re_scores, meta_scores, ide_scores], dim=1)
    dominant = score_matrix.argmax(dim=1)

    rows: list[dict] = []
    kernels: list[CognitiveKernel] = []

    for layer_idx in range(len(kl_profile)):
        module = COGNITIVE_MODULES[int(dominant[layer_idx])]
        layer_scores = {
            "identity": float(id_scores[layer_idx]),
            "reasoning": float(re_scores[layer_idx]),
            "meta_reasoning": float(meta_scores[layer_idx]),
            "ideation": float(ide_scores[layer_idx]),
        }
        top_intent = ""
        if layer_idx < len(trace_a.layer_stats):
            top_intent = str(trace_a.layer_stats.loc[layer_idx, "top1_intent"])

        rows.append(
            {
                "layer": layer_idx,
                "dominant_module": module,
                "kl_divergence": float(kl_profile[layer_idx]),
                "barlow_invariance": float(barlow_diag[layer_idx]),
                "barlow_redundancy": float(barlow_off[layer_idx]),
                **layer_scores,
            }
        )

        # One kernel per module at its peak layer
        if layer_scores[module] == max(
            score_matrix[:, COGNITIVE_MODULES.index(module)]
        ):
            extractor = _MODULE_EXTRACTORS[module]
            kernels.append(
                extractor(
                    layer_idx=layer_idx,
                    kernel_basis=master_basis,
                    kl_weight=float(kl_profile[layer_idx]),
                    barlow_invariance=float(barlow_diag[layer_idx]),
                    barlow_redundancy=float(barlow_off[layer_idx]),
                    score=layer_scores[module],
                    top_intent=top_intent,
                )
            )

    # Ensure one kernel per module (fallback: argmax per column)
    existing = {k.module for k in kernels}
    for i, module in enumerate(COGNITIVE_MODULES):
        if module in existing:
            continue
        best_layer = int(score_matrix[:, i].argmax())
        extractor = _MODULE_EXTRACTORS[module]
        top_intent = str(trace_a.layer_stats.loc[best_layer, "top1_intent"]) if best_layer < len(trace_a.layer_stats) else ""
        kernels.append(
            extractor(
                layer_idx=best_layer,
                kernel_basis=master_basis,
                kl_weight=float(kl_profile[best_layer]),
                barlow_invariance=float(barlow_diag[best_layer]),
                barlow_redundancy=float(barlow_off[best_layer]),
                score=float(score_matrix[best_layer, i]),
                top_intent=top_intent,
            )
        )

    return CognitiveModuleProfile(
        twin_a=twin_a,
        twin_b=twin_b,
        kl_profile=[float(x) for x in kl_profile],
        barlow_metrics=barlow_metrics,
        kernels=kernels,
        layer_assignments=pd.DataFrame(rows),
        projector=projector,
    )


def enrich_layer_map_with_cognitive_modules(
    layer_map: pd.DataFrame,
    cognitive: CognitiveModuleProfile,
) -> pd.DataFrame:
    """Merge cognitive module assignments into the layer correspondence map."""
    if cognitive.layer_assignments.empty:
        return layer_map
    merged = layer_map.merge(
        cognitive.layer_assignments[
            [
                "layer",
                "dominant_module",
                "kl_divergence",
                "barlow_invariance",
                "barlow_redundancy",
                "identity",
                "reasoning",
                "meta_reasoning",
                "ideation",
            ]
        ],
        on="layer",
        how="left",
    )
    return merged
