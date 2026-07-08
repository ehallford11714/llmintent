"""CoT delta profile: per-layer twin shift magnitude for meta-reasoning."""

from __future__ import annotations

import torch

from llmintent.kernels.kl_kernel import collect_twin_hidden_matrix, per_layer_kl_profile
from llmintent.models import ModelBundle


def compute_cot_delta(
    bundle: ModelBundle,
    twin_a: str,
    twin_b: str,
    *,
    position: int = -1,
    kl_weighted: bool = True,
) -> torch.Tensor:
    """
    Per-layer magnitude of hidden-state shift between twin prompts.

    Used to detect layers where chain-of-thought restructuring occurs —
    the signal meta_reasoning_layer_scores expects via cot_delta.
    """
    kl_profile, _ = per_layer_kl_profile(bundle, twin_a, twin_b, position=position)
    h_a, h_b = collect_twin_hidden_matrix(bundle, twin_a, twin_b, position=position)
    delta_norm = torch.norm(h_b - h_a, dim=-1)
    if kl_weighted:
        kl_norm = kl_profile / (kl_profile.max() + 1e-8)
        return delta_norm * (0.5 + 0.5 * kl_norm)
    return delta_norm
