"""SVD decomposition of FFN weights (notebook perform_svd_on_ffn)."""

from __future__ import annotations

import torch


def perform_svd_on_ffn(weight_matrix: torch.Tensor, top_k: int = 50) -> torch.Tensor:
    """
    Decompose FFN weights and return top hidden-dimension components.

    Returns shape [hidden_dim, top_k].
    """
    w = weight_matrix.to(torch.float32)
    if w.shape[0] < w.shape[1]:
        w = w.t()
    _, _, vh = torch.linalg.svd(w, full_matrices=False)
    return vh[:top_k, :].t()


def top_projected_components(
    weight_matrix: torch.Tensor,
    projection: torch.Tensor,
    top_k: int = 50,
) -> torch.Tensor:
    """SVD components projected into GloVe space. Shape [top_k, glove_dim]."""
    top_v = perform_svd_on_ffn(weight_matrix, top_k=top_k)
    return top_v.t() @ projection
