"""Meta-reasoning kernel: monitoring / restructuring under twin KL spikes."""

from __future__ import annotations

import torch


def meta_reasoning_layer_scores(
    kl_profile: torch.Tensor,
    barlow_offdiag: torch.Tensor,
    *,
    cot_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Score layers for meta-reasoning (thinking about thinking).

    High when KL divergence spikes (twin mismatch) and Barlow off-diagonal
    redundancy is elevated (cross-feature re-combination).
    """
    kl_norm = kl_profile / (kl_profile.max() + 1e-8)
    off_norm = barlow_offdiag / (barlow_offdiag.max() + 1e-8)
    scores = kl_norm * off_norm
    if cot_delta is not None:
        delta_norm = cot_delta / (cot_delta.max() + 1e-8)
        scores = scores * (0.5 + 0.5 * delta_norm)
    return scores


def extract_meta_reasoning_kernel(
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
        module="meta_reasoning",
        layer=layer_idx,
        score=score,
        kl_weight=kl_weight,
        barlow_invariance=barlow_invariance,
        barlow_redundancy=barlow_redundancy,
        kernel_basis=kernel_basis,
        top_intent=top_intent,
    )
