"""Reference poles for semantic steering analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from llmintent.embeddings import EmbeddingSpace
from llmintent.models import ModelBundle, get_input_embeddings


@dataclass
class ReferencePoles:
    semantic: torch.Tensor
    grammatical: torch.Tensor
    numerical: torch.Tensor | None = None


def build_glove_poles(embedding_space: EmbeddingSpace) -> ReferencePoles:
    """Build semantic and grammatical poles in GloVe space."""
    semantic = torch.from_numpy(embedding_space.mean_vector(limit=10_000))
    gram_tokens = [".", ",", "and", "but", "in", "of", "to", "the"]
    gram_vecs = [
        embedding_space.vectors[embedding_space.index(t)]
        for t in gram_tokens
        if embedding_space.index(t) is not None
    ]
    grammatical = torch.from_numpy(np.array(gram_vecs).mean(axis=0))
    return ReferencePoles(semantic=semantic, grammatical=grammatical)


def build_numerical_pole(bundle: ModelBundle, tokens: list[str] | None = None) -> torch.Tensor:
    """Build a numerical pole from token embedding averages (GPT-style)."""
    num_tokens = tokens or ["0", "1", "2", "3", "4", "5", "sum", "total", "equal"]
    embeddings = get_input_embeddings(bundle.model)
    ids: list[int] = []
    for token in num_tokens:
        encoded = bundle.tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    if not ids:
        raise ValueError("No single-token numerical anchors found in vocabulary")
    return torch.mean(embeddings[ids].float(), dim=0)
