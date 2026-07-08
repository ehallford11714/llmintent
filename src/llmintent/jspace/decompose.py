"""Sparse J-space decomposition into active verbal intents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llmintent.forward import normalize_hidden
from llmintent.models import ModelBundle, get_unembedding_matrix


@dataclass(frozen=True)
class SparseIntent:
    token_id: int
    token: str
    weight: float


def sparse_intent_decomposition(
    bundle: ModelBundle,
    hidden: torch.Tensor,
    *,
    k: int = 16,
    transport: torch.Tensor | None = None,
) -> list[SparseIntent]:
    """
    Greedy sparse nonnegative decomposition over unembedding rows (J-space proxy).

    Mirrors Anthropic's sparse verbal intent extraction (gradient pursuit simplified).
    """
    unembed = get_unembedding_matrix(bundle.model).float()
    h = hidden.float()
    if transport is not None:
        h = h @ transport.to(h.device).T
    target = normalize_hidden(bundle, h)

    # Nonnegative matching pursuit on unembedding dictionary
    dictionary = F.normalize(unembed, dim=1)
    residual = target.clone()
    active: list[tuple[int, float]] = []
    used: set[int] = set()

    for _ in range(k):
        scores = dictionary @ residual
        scores[list(used)] = -1.0
        best_id = int(scores.argmax().item())
        if scores[best_id] <= 0:
            break
        direction = dictionary[best_id]
        coef = float(torch.dot(residual, direction).clamp(min=0).item())
        if coef <= 1e-8:
            break
        active.append((best_id, coef))
        used.add(best_id)
        residual = residual - coef * direction

    active.sort(key=lambda x: x[1], reverse=True)
    return [
        SparseIntent(
            token_id=tid,
            token=bundle.tokenizer.decode([tid]).strip(),
            weight=weight,
        )
        for tid, weight in active
    ]


def jspace_occupancy(
    bundle: ModelBundle,
    hidden: torch.Tensor,
    *,
    max_k: int = 25,
    transport: torch.Tensor | None = None,
    threshold: float = 0.01,
) -> int:
    """Count active J-space concepts before marginal gain drops (Anthropic occupancy metric)."""
    sparse = sparse_intent_decomposition(
        bundle,
        hidden,
        k=max_k,
        transport=transport,
    )
    return sum(1 for s in sparse if s.weight >= threshold)
