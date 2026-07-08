"""Identity kernel: stable self-aligned representation (low KL, high Barlow diagonal)."""

from __future__ import annotations

import torch


def identity_layer_scores(
    kl_profile: torch.Tensor,
    barlow_diag: torch.Tensor,
    *,
    motor_alignment: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Score layers for identity binding.

    High when KL is low (twin-stable) and Barlow diagonal is high (invariant).
    """
    kl_norm = kl_profile / (kl_profile.max() + 1e-8)
    stability = 1.0 - kl_norm
    scores = stability * barlow_diag
    if motor_alignment is not None:
        # Identity layers precede motor readout
        scores = scores * (1.0 - motor_alignment)
    return scores


def extract_identity_kernel(
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
        module="identity",
        layer=layer_idx,
        score=score,
        kl_weight=kl_weight,
        barlow_invariance=barlow_invariance,
        barlow_redundancy=barlow_redundancy,
        kernel_basis=kernel_basis,
        top_intent=top_intent,
    )
