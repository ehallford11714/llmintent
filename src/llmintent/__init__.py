"""LLMIntent — semantic extraction and intent analysis for transformer LLMs."""

from llmintent.activation import activation_summary, identify_activation_layers
from llmintent.analyzer import AnalysisReport, LLMIntentAnalyzer
from llmintent.compaction import CompactionAnalyzer
from llmintent.cognitive import CognitiveModuleProfile, build_cognitive_module_profile
from llmintent.jspace import (
    IntentTrace,
    TransportMaps,
    build_intent_trace,
    classify_layer_regimes,
    fit_transport_maps,
)
from llmintent.kernels import minimize_twin_barlow, per_layer_kl_profile
from llmintent.query import ConceptQueryResult, query_concept_in_trajectory, query_concepts_batch
from llmintent.layers import build_layer_correspondence_map, summarize_layer_bands
from llmintent.metrics import calculate_sso_score, kl_divergence, shannon_entropy

__all__ = [
    "AnalysisReport",
    "CognitiveModuleProfile",
    "CompactionAnalyzer",
    "ConceptQueryResult",
    "IntentTrace",
    "LLMIntentAnalyzer",
    "TransportMaps",
    "activation_summary",
    "build_cognitive_module_profile",
    "build_intent_trace",
    "build_layer_correspondence_map",
    "calculate_sso_score",
    "classify_layer_regimes",
    "fit_transport_maps",
    "identify_activation_layers",
    "kl_divergence",
    "minimize_twin_barlow",
    "per_layer_kl_profile",
    "query_concept_in_trajectory",
    "query_concepts_batch",
    "shannon_entropy",
    "summarize_layer_bands",
]

__version__ = "0.4.0"
