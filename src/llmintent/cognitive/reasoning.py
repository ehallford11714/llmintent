"""Reasoning kernel: workspace computation under KL tension."""

from __future__ import annotations

import torch


def reasoning_layer_scores(
    kl_profile: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    entropy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Score layers for primary reasoning.

    Peaks where KL tension is moderate-high and J-space occupancy is high.
    """
    kl_norm = kl_profile / (kl_profile.max() + 1e-8)
    occ_norm = occupancy / (occupancy.max() + 1e-8)
    # Bell-shaped preference for mid-high KL (not identity, not pure motor)
    kl_peak = kl_norm * (1.0 - kl_norm)
    scores = kl_peak * occ_norm
    if entropy is not None:
        ent_norm = entropy / (entropy.max() + 1e-8)
        scores = scores * (0.5 + 0.5 * ent_norm)
    return scores


def extract_reasoning_kernel(
    layer_idx: int,
    kernel_basis: torch.Tensor,
    kl_weight: float,
    barlow_invariance: float,
    barlow_redundancy: float,
    score: float,
    top_intent: str = "",
):
    from llmintent.cognitive.types import CognitiveKernel

    return CognitiveKernel(
        module="reasoning",
        layer=layer_idx,
        score=score,
        kl_weight=kl_weight,
        barlow_invariance=barlow_invariance,
        barlow_redundancy=barlow_redundancy,
        kernel_basis=kernel_basis,
        top_intent=top_intent,
    )
