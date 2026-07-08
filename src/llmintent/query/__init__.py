"""Semantic concept queries against activation trajectories."""

from llmintent.query.concept_query import (
    ConceptQueryResult,
    query_concept_in_trajectory,
    query_concepts_batch,
)
from llmintent.query.feature_space import (
    build_trajectory_feature_space,
    semantic_concept_vector,
)

__all__ = [
    "ConceptQueryResult",
    "build_trajectory_feature_space",
    "query_concept_in_trajectory",
    "query_concepts_batch",
    "semantic_concept_vector",
]
