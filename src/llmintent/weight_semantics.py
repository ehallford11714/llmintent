"""Map weight slices to vocabulary via KNN (notebook weight semantics)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from llmintent.embeddings import EmbeddingSpace
from llmintent.morphemes import MorphemeExtractor


def knn_words_for_weight_slice(
    weight_slice: np.ndarray,
    embedding_space: EmbeddingSpace,
    *,
    top_n: int = 100,
    vocab_limit: int = 50_000,
) -> list[str]:
    """Find vocabulary words nearest to a weight slice mean in embedding space."""
    query = np.full((1, embedding_space.dim), float(weight_slice.mean()))
    vectors = embedding_space.vectors[:vocab_limit]
    vocab = embedding_space.vocab[:vocab_limit]
    nn = NearestNeighbors(n_neighbors=min(top_n, len(vocab)), metric="cosine")
    nn.fit(vectors)
    _, indices = nn.kneighbors(query)
    return [vocab[i] for i in indices[0]]


def morpheme_experiment(
    weight_slice: np.ndarray,
    embedding_space: EmbeddingSpace,
    morpheme_freq: Counter[str],
    *,
    extractor: MorphemeExtractor | None = None,
    top_n: int = 100,
) -> list[str]:
    """Notebook: stanza_morpheme_experiment / updated_morpheme_experiment."""
    extractor = extractor or MorphemeExtractor("lemma")
    matching_words = knn_words_for_weight_slice(weight_slice, embedding_space, top_n=top_n)
    found_units = extractor.extract(matching_words)
    unique = list(dict.fromkeys(found_units))
    ranked = sorted(unique, key=lambda x: morpheme_freq.get(x, 0), reverse=True)
    return ranked[:20]


def extract_semantic_well(
    weight_slice: np.ndarray,
    embedding_space: EmbeddingSpace,
    *,
    extractor: MorphemeExtractor | None = None,
    top_k: int = 25,
) -> list[str]:
    """Notebook: extract_semantic_well — themed morpheme clusters for a weight block."""
    extractor = extractor or MorphemeExtractor("lemma")
    matching_words = knn_words_for_weight_slice(weight_slice, embedding_space, top_n=top_k)
    themes = extractor.top_themes(matching_words, top_k=5)
    return [f"{theme} ({count})" for theme, count in themes]


def get_block_semantics(
    layer: Any,
    embedding_space: EmbeddingSpace,
    morpheme_freq: Counter[str],
    *,
    extractor: MorphemeExtractor | None = None,
    max_rows: int = 70,
    max_cols: int = 70,
) -> dict[str, list[str]]:
    """Notebook: get_block_semantics for DistilBERT-style layers."""
    components = {}
    if hasattr(layer, "attention") and hasattr(layer, "ffn"):
        components = {
            "Attention Query": layer.attention.q_lin.weight.data.cpu().numpy()[:max_rows, :max_cols],
            "Attention Key": layer.attention.k_lin.weight.data.cpu().numpy()[:max_rows, :max_cols],
            "Attention Value": layer.attention.v_lin.weight.data.cpu().numpy()[:max_rows, :max_cols],
            "Attention Output": layer.attention.out_lin.weight.data.cpu().numpy()[:max_rows, :max_cols],
            "FFN Intermediate": layer.ffn.lin1.weight.data.cpu().numpy()[:max_rows, :max_cols],
            "FFN Output": layer.ffn.lin2.weight.data.cpu().numpy()[:max_rows, :max_cols],
        }
    elif hasattr(layer, "mlp"):
        w = layer.mlp.up_proj.weight.data.cpu().numpy() if hasattr(layer.mlp, "up_proj") else layer.mlp.c_fc.weight.data.cpu().numpy()
        components = {"FFN Up": w[:max_rows, :max_cols]}
    else:
        raise AttributeError("Unsupported layer structure for block semantics")

    block_map: dict[str, list[str]] = {}
    for name, weight_slice in components.items():
        block_map[name] = morpheme_experiment(
            weight_slice,
            embedding_space,
            morpheme_freq,
            extractor=extractor,
            top_n=50,
        )[:10]
    return block_map


def get_block_expertise(semantics: dict[str, list[str]]) -> str:
    """Heuristic expertise label from notebook."""
    all_units = [unit for sublist in semantics.values() for unit in sublist]
    counts = Counter(all_units)
    if counts.get("kilometer") or counts.get("park") or counts.get("east"):
        return "Spatial & Physical Navigation"
    if counts.get("recommend") or counts.get("draft") or counts.get("economist"):
        return "Strategic & Deliberative Logic"
    return "General Semantic Transformation"
