"""Tests for KL-Barlow-KNN concept query (synthetic, no model load)."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def test_knn_finds_nearest_layer_in_barlow_space():
    proj_dim = 8
    n_layers = 12
    rng = np.random.default_rng(0)

    # Layer features: random + one layer strongly aligned to concept
    layer_feats = rng.standard_normal((n_layers, proj_dim)).astype(np.float32)
    concept = rng.standard_normal(proj_dim).astype(np.float32)
    concept = concept / np.linalg.norm(concept)
    layer_feats[7] = concept + 0.01 * rng.standard_normal(proj_dim)

    norms = np.linalg.norm(layer_feats, axis=1, keepdims=True)
    layer_feats = layer_feats / np.clip(norms, 1e-8, None)

    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(layer_feats)
    dists, idx = nn.kneighbors(concept.reshape(1, -1))

    assert 7 in idx[0]
    assert dists[0][0] < 0.1


def test_fused_score_formula():
    knn_sim = 0.9
    kl_w = 0.8
    barlow_inv = 0.7
    probe = 0.6
    fused = knn_sim * (0.4 + 0.6 * kl_w) * (0.5 + 0.5 * barlow_inv) * (0.3 + 0.7 * probe)
    assert 0.3 < fused < 1.0


def test_trajectory_concept_similarity_shape():
    n = 6
    features = np.random.default_rng(1).standard_normal((n, 11)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    concept_vec = features[3]  # perfect match at layer 3
    sims = features @ concept_vec
    assert sims.argmax() == 3
