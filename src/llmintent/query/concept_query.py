"""Query feature-space concepts against activation trajectories (KL + Barlow + KNN)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from llmintent.activation import activation_summary, identify_activation_layers
from llmintent.cognitive.types import CognitiveModuleProfile
from llmintent.jspace.decode import probe_concept
from llmintent.forward import forward_hidden_states
from llmintent.models import ModelBundle
from llmintent.query.feature_space import (
    build_trajectory_feature_space,
    semantic_concept_vector,
)


@dataclass
class ConceptQueryResult:
    """Result of querying a semantic concept against an activation trajectory."""

    concept: str
    prompt: str
    twin_b: str | None
    peak_layer: int
    matched_layers: list[int]
    trajectory: pd.DataFrame
    knn_ranking: pd.DataFrame
    concept_diagnostics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "prompt": self.prompt,
            "twin_b": self.twin_b,
            "peak_layer": self.peak_layer,
            "matched_layers": self.matched_layers,
            "concept_diagnostics": self.concept_diagnostics,
            "knn_ranking": self.knn_ranking.to_dict(orient="records"),
            "trajectory": self.trajectory.to_dict(orient="records"),
        }


def query_concept_in_trajectory(
    bundle: ModelBundle,
    concept: str,
    prompt: str,
    *,
    twin_b: str | None = None,
    position: int = -1,
    top_k_layers: int = 5,
    knn_neighbors: int = 3,
    proj_dim: int = 32,
    cognitive_profile: CognitiveModuleProfile | None = None,
) -> ConceptQueryResult:
    """
    Query a semantic concept against the activation trajectory using
    KL + twin Barlow feature space and KNN retrieval.

    Strategy:
    1. Build KL-weighted Barlow layer feature matrix from twin prompts
    2. Embed concept text into the same feature space
    3. KNN (cosine) retrieves nearest activation layers
    4. Re-rank by KL × Barlow × semantic probe for peak activation
    """
    twin_b = twin_b or prompt
    features, meta, projector = build_trajectory_feature_space(
        bundle,
        prompt,
        twin_b,
        position=position,
        proj_dim=proj_dim,
    )
    concept_vec, diagnostics = semantic_concept_vector(
        bundle,
        concept,
        projector,
        anchor_prompt=prompt,
    )

    n_layers = features.shape[0]
    proj_dim = projector.shape[1]
    layer_knn_feats = features[:, :proj_dim]
    concept_knn = concept_vec[:proj_dim]

    k = min(knn_neighbors, n_layers)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(layer_knn_feats)
    distances, indices = nn.kneighbors(concept_knn.reshape(1, -1))

    # Semantic probe scores per layer on the anchor prompt
    _, states = forward_hidden_states(bundle, prompt)
    pos = position if position >= 0 else states[0].shape[1] + position
    probe_scores = _concept_probe_layers(bundle, concept, states, pos)

    activation = identify_activation_layers(bundle, prompt, position=pos)
    act_df = activation_summary(bundle, prompt, position=pos)

    rows: list[dict] = []
    for rank, (layer_idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        layer = int(layer_idx)
        kl_w = float(meta.loc[layer, "kl_weight"])
        barlow_inv = float(meta.loc[layer, "barlow_invariance"])
        barlow_red = float(meta.loc[layer, "barlow_redundancy"])
        probe = float(probe_scores[layer])
        knn_sim = 1.0 - float(dist)
        # Fused activation score: KNN similarity × KL tension × Barlow structure × probe
        fused = knn_sim * (0.4 + 0.6 * kl_w) * (0.5 + 0.5 * barlow_inv) * (0.3 + 0.7 * probe)

        dominant_module = ""
        if cognitive_profile is not None and not cognitive_profile.layer_assignments.empty:
            match = cognitive_profile.layer_assignments[
                cognitive_profile.layer_assignments["layer"] == layer
            ]
            if not match.empty:
                dominant_module = str(match.iloc[0]["dominant_module"])

        pivot_tags = [name for name, idx in activation.items() if idx == layer]

        rows.append(
            {
                "rank": rank,
                "layer": layer,
                "knn_distance": float(dist),
                "knn_similarity": knn_sim,
                "kl_weight": kl_w,
                "barlow_invariance": barlow_inv,
                "barlow_redundancy": barlow_red,
                "semantic_probe": probe,
                "fused_activation_score": fused,
                "dominant_module": dominant_module,
                "is_activation_pivot": bool(pivot_tags),
                "pivot_tags": ", ".join(pivot_tags),
            }
        )

    knn_df = pd.DataFrame(rows).sort_values("fused_activation_score", ascending=False)
    knn_df["rank"] = range(1, len(knn_df) + 1)

    peak_layer = int(knn_df.iloc[0]["layer"])
    matched_layers = knn_df["layer"].astype(int).head(top_k_layers).tolist()

    trajectory = _build_annotated_trajectory(
        meta,
        act_df,
        probe_scores,
        features,
        concept_vec,
        cognitive_profile,
        activation,
    )

    return ConceptQueryResult(
        concept=concept,
        prompt=prompt,
        twin_b=twin_b,
        peak_layer=peak_layer,
        matched_layers=matched_layers,
        trajectory=trajectory,
        knn_ranking=knn_df,
        concept_diagnostics=diagnostics,
    )


def query_concepts_batch(
    bundle: ModelBundle,
    concepts: list[str],
    prompt: str,
    *,
    twin_b: str | None = None,
    **kwargs,
) -> dict[str, ConceptQueryResult]:
    """Query multiple semantic concepts against the same activation trajectory."""
    return {
        concept: query_concept_in_trajectory(
            bundle, concept, prompt, twin_b=twin_b, **kwargs
        )
        for concept in concepts
    }


def _concept_probe_layers(
    bundle: ModelBundle,
    concept: str,
    states: list,
    position: int,
) -> list[float]:
    """Max J-space probe score across concept tokens at each layer."""
    tokens = concept.lower().split()
    scores: list[float] = []
    for state in states:
        hidden = state[0, position, :]
        if not tokens:
            scores.append(0.0)
            continue
        token_scores = [probe_concept(bundle, hidden, t) for t in tokens]
        scores.append(max(token_scores) if token_scores else 0.0)
    return scores


def _build_annotated_trajectory(
    meta: pd.DataFrame,
    act_df: pd.DataFrame,
    probe_scores: list[float],
    features: np.ndarray,
    concept_vec: np.ndarray,
    cognitive_profile: CognitiveModuleProfile | None,
    activation: dict[str, int],
) -> pd.DataFrame:
    """Full layer trajectory with concept relevance annotated."""
    pivot_layers = set(activation.values())
    concept_sims = features @ concept_vec  # cosine since both normalized

    traj = meta.merge(act_df, on="layer", how="left")
    traj["semantic_probe"] = probe_scores
    traj["concept_similarity"] = concept_sims
    traj["is_activation_pivot"] = traj["layer"].isin(pivot_layers)

    if cognitive_profile is not None and not cognitive_profile.layer_assignments.empty:
        traj = traj.merge(
            cognitive_profile.layer_assignments[
                ["layer", "dominant_module", "reasoning", "meta_reasoning", "ideation"]
            ],
            on="layer",
            how="left",
        )

    traj["concept_activation"] = (
        traj["concept_similarity"]
        * (0.4 + 0.6 * traj["kl_weight"])
        * (0.3 + 0.7 * traj["semantic_probe"])
    )
    return traj.sort_values("layer").reset_index(drop=True)
