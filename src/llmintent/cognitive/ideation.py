"""Ideation kernel: divergent generation away from motor readout."""

from __future__ import annotations

import torch


def ideation_layer_scores(
    entropy: torch.Tensor,
    motor_alignment: torch.Tensor,
    *,
    kl_profile: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Score layers for ideation / creative divergence.

    High entropy, low motor alignment — representations exploring before commit.
    """
    ent_norm = entropy / (entropy.max() + 1e-8)
    diverge = 1.0 - motor_alignment
    scores = ent_norm * diverge
    if kl_profile is not None:
        kl_norm = kl_profile / (kl_profile.max() + 1e-8)
        # Ideation tolerates moderate KL spread
        scores = scores * (0.3 + 0.7 * kl_norm)
    return scores


def extract_ideation_kernel(
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
        module="ideation",
        layer=layer_idx,
        score=score,
        kl_weight=kl_weight,
        barlow_invariance=barlow_invariance,
        barlow_redundancy=barlow_redundancy,
        kernel_basis=kernel_basis,
        top_intent=top_intent,
    )
