from collections import Counter

import numpy as np

from llmintent.embeddings import EmbeddingSpace
from llmintent.morphemes import MorphemeExtractor
from llmintent.weight_semantics import extract_semantic_well, morpheme_experiment


def test_lemma_extractor():
    ext = MorphemeExtractor("lemma")
    assert ext.extract(["Running", "Dogs"]) == ["running", "dogs"]


def test_semantic_well_with_toy_embeddings():
    space = EmbeddingSpace(
        vocab=["dog", "cat", "run", "jump", "the"],
        vectors=np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32),
    )
    weight = np.zeros((4, 4))
    themes = extract_semantic_well(weight, space, extractor=MorphemeExtractor("lemma"))
    assert len(themes) <= 5


def test_morpheme_experiment_ranking():
    space = EmbeddingSpace(
        vocab=["dog", "cat", "run"],
        vectors=np.array([[1, 0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
    )
    freq = Counter({"dog": 10, "cat": 5, "run": 1})
    units = morpheme_experiment(np.ones((2, 2)), space, freq, top_n=2)
    assert isinstance(units, list)
