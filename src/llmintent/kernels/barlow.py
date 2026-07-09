"""Twin Barlow Twins cross-correlation and minimization for kernel discovery."""

from __future__ import annotations

import torch
import torch.nn as nn


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_cross_correlation(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Cross-correlation matrix C where C_ij = corr(z_a_i, z_b_j)."""
    za = _batch_norm(z_a, eps)
    zb = _batch_norm(z_b, eps)
    n = za.shape[0]
    return (za.T @ zb) / max(n, 1)


def barlow_twins_loss(
    cross_corr: torch.Tensor,
    *,
    lambda_coeff: float = 0.005,
) -> torch.Tensor:
    """Barlow Twins loss: diagonal → 1, off-diagonal → 0."""
    on_diag = torch.diagonal(cross_corr).add(-1.0).pow(2).sum()
    off_diag = _off_diagonal(cross_corr).pow(2).sum()
    return on_diag + lambda_coeff * off_diag


def barlow_invariance_score(cross_corr: torch.Tensor) -> float:
    return float(torch.diagonal(cross_corr).mean().item())


def barlow_redundancy_score(cross_corr: torch.Tensor) -> float:
    return float(_off_diagonal(cross_corr).pow(2).mean().item())


def per_layer_barlow_features(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-layer Barlow diagonal and off-diagonal energy.

    Treats each layer as one sample; returns [n_layers] tensors.
    """
    n_layers = h_a.shape[0]
    diag = torch.zeros(n_layers)
    off = torch.zeros(n_layers)
    for i in range(n_layers):
        c = barlow_cross_correlation(h_a[i : i + 1], h_b[i : i + 1])
        diag[i] = torch.diagonal(c).mean()
        off[i] = _off_diagonal(c).pow(2).mean() if c.shape[0] > 1 else 0.0
    # Global layer-context cross-correlation for richer off-diagonal structure
    c_full = barlow_cross_correlation(h_a, h_b)
    off_global = _off_diagonal(c_full).pow(2).mean()
    off = off + off_global
    return diag, off


def minimize_twin_barlow(
    h_twin_a: torch.Tensor,
    h_twin_b: torch.Tensor,
    kl_weights: torch.Tensor,
    *,
    proj_dim: int = 32,
    steps: int = 80,
    lr: float = 0.05,
    lambda_coeff: float = 0.005,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Learn kernel projector W minimizing Barlow loss on KL-weighted twin states."""
    if h_twin_a.shape != h_twin_b.shape:
        raise ValueError("Twin hidden tensors must have the same shape")

    # Detach from any prior forward graph — safe for repeated live / heighten calls.
    h_twin_a = h_twin_a.detach().float()
    h_twin_b = h_twin_b.detach().float()
    kl_weights = kl_weights.detach().float()

    hidden_dim = h_twin_a.shape[1]
    proj_dim = min(proj_dim, hidden_dim, h_twin_a.shape[0])

    w = nn.Parameter(torch.randn(hidden_dim, proj_dim, dtype=h_twin_a.dtype) * 0.02)
    opt = torch.optim.Adam([w], lr=lr)
    kl = kl_weights.float()
    if kl.sum() <= 0:
        kl = torch.ones_like(kl)
    kl = (kl / kl.sum()).detach()

    last_loss = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        za = h_twin_a @ w
        zb = h_twin_b @ w
        za = za * kl.unsqueeze(-1)
        zb = zb * kl.unsqueeze(-1)
        c = barlow_cross_correlation(za, zb)
        loss = barlow_twins_loss(c, lambda_coeff=lambda_coeff)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    with torch.no_grad():
        c_final = barlow_cross_correlation(h_twin_a @ w, h_twin_b @ w)

    metrics = {
        "barlow_loss": last_loss,
        "invariance": barlow_invariance_score(c_final),
        "redundancy": barlow_redundancy_score(c_final),
    }
    return w.detach(), metrics


def extract_kernel_basis(projector: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(projector, mode="reduced")
    return q.T


def _batch_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp(min=eps)
    return (x - mean) / std
