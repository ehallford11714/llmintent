"""Model ↔ GloVe projection (least-squares alignment from the notebook)."""

from __future__ import annotations

import torch

from llmintent.embeddings import EmbeddingSpace
from llmintent.models import get_input_embeddings


def build_projection_matrix(
    model,
    tokenizer,
    embedding_space: EmbeddingSpace,
    *,
    common_limit: int = 3000,
) -> torch.Tensor:
    """
    Solve X @ P = Y where X is model token embeddings and Y is GloVe vectors.

    Returns P with shape [hidden_dim, glove_dim].
    """
    model_embeddings = get_input_embeddings(model)
    model_vecs: list[torch.Tensor] = []
    glove_subset: list[list[float]] = []
    count = 0

    for i, word in enumerate(embedding_space.vocab):
        if count >= common_limit:
            break
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        model_vecs.append(model_embeddings[token_ids[0]].detach().cpu())
        glove_subset.append(embedding_space.vectors[i].tolist())
        count += 1

    if not model_vecs:
        raise ValueError("No single-token overlap between tokenizer and GloVe vocabulary")

    x = torch.stack(model_vecs).to(torch.float32)
    y = torch.tensor(glove_subset, dtype=torch.float32)
    return torch.linalg.lstsq(x, y).solution


def project_hidden_to_glove(
    hidden_vectors: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Project hidden-dim vectors into GloVe space via P."""
    return hidden_vectors.float() @ projection
