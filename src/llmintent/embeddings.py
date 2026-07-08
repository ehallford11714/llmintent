"""GloVe / Word2Vec loading for semantic projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingSpace:
    """Vocabulary-aligned dense embedding matrix."""

    vocab: list[str]
    vectors: np.ndarray  # shape [N, dim]

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def index(self, word: str) -> int | None:
        try:
            return self.vocab.index(word)
        except ValueError:
            return None

    def vector_for(self, word: str) -> np.ndarray | None:
        idx = self.index(word)
        if idx is None:
            return None
        return self.vectors[idx]

    def mean_vector(self, limit: int | None = None) -> np.ndarray:
        subset = self.vectors[:limit] if limit else self.vectors
        return subset.mean(axis=0)


def load_glove_gensim(name: str = "glove-wiki-gigaword-100") -> EmbeddingSpace:
    """Load a Gensim pre-trained embedding (default: 100-d GloVe)."""
    import gensim.downloader as api

    keyed = api.load(name)
    vocab = list(keyed.index_to_key)
    vectors = np.stack([keyed[w] for w in vocab], axis=0).astype(np.float32)
    return EmbeddingSpace(vocab=vocab, vectors=vectors)


def load_glove_text(path: str) -> EmbeddingSpace:
    """Load GloVe vectors from a plain-text file (word dim val1 val2 ...)."""
    vocab: list[str] = []
    rows: list[list[float]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            vocab.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    return EmbeddingSpace(vocab=vocab, vectors=np.array(rows, dtype=np.float32))
