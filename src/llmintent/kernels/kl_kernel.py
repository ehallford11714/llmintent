"""KL-weighted kernel identification between twin prompt trajectories."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmintent.forward import forward_hidden_states, get_lm_head, normalize_hidden
from llmintent.metrics import kl_divergence
from llmintent.models import ModelBundle


def per_layer_kl_profile(
    bundle: ModelBundle,
    twin_a: str,
    twin_b: str,
    *,
    position: int = -1,
) -> tuple[torch.Tensor, list[int]]:
    """
    KL(P_b || P_a) at each layer for the final token distribution.

    Returns kl_weights [n_layers] and layer indices.
    """
    _, states_a = forward_hidden_states(bundle, twin_a)
    _, states_b = forward_hidden_states(bundle, twin_b)
    head = get_lm_head(bundle)
    values: list[float] = []

    for sa, sb in zip(states_a, states_b):
        pos = position if position >= 0 else sa.shape[1] + position
        logits_a = head(sa[0, pos, :])
        logits_b = head(sb[0, pos, :])
        pa = F.softmax(logits_a, dim=-1)
        pb = F.softmax(logits_b, dim=-1)
        values.append(kl_divergence(pb, pa))

    kl = torch.tensor(values, dtype=torch.float32)
    return kl, list(range(len(values)))


def collect_twin_hidden_matrix(
    bundle: ModelBundle,
    twin_a: str,
    twin_b: str,
    *,
    position: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-layer hidden states for twin prompts. Shape [n_layers, hidden]."""
    _, states_a = forward_hidden_states(bundle, twin_a)
    _, states_b = forward_hidden_states(bundle, twin_b)
    pos = position if position >= 0 else states_a[0].shape[1] + position

    with torch.no_grad():
        rows_a = [
            normalize_hidden(bundle, s[0, pos, :].float()).detach().cpu()
            for s in states_a
        ]
        rows_b = [
            normalize_hidden(bundle, s[0, pos, :].float()).detach().cpu()
            for s in states_b
        ]
    return torch.stack(rows_a), torch.stack(rows_b)


def kl_weighted_difference_kernel(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    kl_weights: torch.Tensor,
    *,
    top_k: int = 8,
) -> torch.Tensor:
    """
    SVD kernel from KL-weighted twin difference.

    Returns top-k left singular vectors [k, hidden_dim].
    """
    diff = (h_b - h_a) * kl_weights.unsqueeze(-1)
    _, _, vh = torch.linalg.svd(diff, full_matrices=False)
    k = min(top_k, vh.shape[0])
    return vh[:k, :]
