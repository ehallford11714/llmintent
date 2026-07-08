"""Metric functions from the SemanticExtractionLLms workbook."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def calculate_sso_score(sem_sim: float, str_sim: float) -> float:
    """
    Semantic-Structural Orthogonality (SSO).

    Formula: (|SemSim| - |StrSim|) / (|SemSim| + |StrSim|)
    """
    abs_sem = abs(sem_sim)
    abs_str = abs(str_sim)
    denominator = abs_sem + abs_str
    if denominator < 1e-12:
        return 0.0
    return (abs_sem - abs_str) / denominator


def shannon_entropy(probs: torch.Tensor) -> float:
    """Shannon entropy of a probability distribution."""
    p = probs.clamp(min=1e-10)
    return float(-torch.sum(p * torch.log(p)).item())


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(P || Q) for two probability distributions."""
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    return float(torch.sum(p * (torch.log(p) - torch.log(q))).item())


def cosine_intensity(vector: torch.Tensor, pole: torch.Tensor) -> float:
    """Absolute cosine similarity between a hidden state and a reference pole."""
    return abs(
        F.cosine_similarity(
            vector.unsqueeze(0).float(),
            pole.unsqueeze(0).float(),
        ).item()
    )


def logits_entropy_from_hidden(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
) -> float:
    logits = lm_head(hidden)
    probs = F.softmax(logits, dim=-1)
    return shannon_entropy(probs)
