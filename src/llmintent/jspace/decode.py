"""Decode hidden states into ranked verbal intents (logit lens + J-lens transport)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llmintent.forward import normalize_hidden
from llmintent.models import ModelBundle


@dataclass(frozen=True)
class IntentToken:
    token_id: int
    token: str
    probability: float
    logit: float
    rank: int


def logit_lens_decode(
    bundle: ModelBundle,
    hidden: torch.Tensor,
    *,
    top_k: int = 10,
) -> list[IntentToken]:
    """
    Standard logit lens: project normalized hidden state through unembedding.

    Equivalent to J-lens with J = I (Anthropic paper baseline).
    """
    from llmintent.models import get_unembedding_matrix

    unembed = get_unembedding_matrix(bundle.model)
    normed = normalize_hidden(bundle, hidden)
    logits = F.linear(normed, unembed)
    return _top_intents(bundle, logits, top_k)


def decode_intents(
    bundle: ModelBundle,
    hidden: torch.Tensor,
    *,
    layer: int,
    transport: torch.Tensor | None = None,
    top_k: int = 10,
) -> list[IntentToken]:
    """
    J-lens decode: softmax(W_U @ norm(J_l @ h_l)).

    When transport is None, falls back to logit lens (identity transport).
    """
    from llmintent.models import get_unembedding_matrix

    unembed = get_unembedding_matrix(bundle.model)
    h = hidden.float()
    if transport is not None:
        h = h @ transport.to(h.device).T
    normed = normalize_hidden(bundle, h)
    logits = F.linear(normed, unembed)
    return _top_intents(bundle, logits, top_k)


def probe_concept(
    bundle: ModelBundle,
    hidden: torch.Tensor,
    token: str,
    *,
    transport: torch.Tensor | None = None,
) -> float:
    """Workspace loading score for a single concept (inner product in logit space)."""
    from llmintent.models import get_unembedding_matrix

    token_ids = bundle.tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        return 0.0
    token_id = token_ids[0]
    unembed = get_unembedding_matrix(bundle.model)
    h = hidden.float()
    if transport is not None:
        h = h @ transport.to(h.device).T
    normed = normalize_hidden(bundle, h)
    logits = F.linear(normed, unembed)
    return float(logits[token_id].item())


def _top_intents(bundle: ModelBundle, logits: torch.Tensor, top_k: int) -> list[IntentToken]:
    probs = F.softmax(logits, dim=-1)
    k = min(top_k, logits.shape[-1])
    top_probs, top_ids = torch.topk(probs, k)
    top_logits = logits[top_ids]
    return [
        IntentToken(
            token_id=int(tid),
            token=bundle.tokenizer.decode([int(tid)]).strip(),
            probability=float(prob.detach().item()),
            logit=float(log.detach().item()),
            rank=i + 1,
        )
        for i, (prob, log, tid) in enumerate(zip(top_probs, top_logits, top_ids))
    ]
