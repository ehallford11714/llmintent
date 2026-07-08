"""LLMIntent — semantic extraction and intent analysis for transformer LLMs."""

from llmintent.activation import activation_summary, identify_activation_layers
from llmintent.analyzer import AnalysisReport, LLMIntentAnalyzer
from llmintent.compaction import CompactionAnalyzer
from llmintent.cognitive import CognitiveModuleProfile, build_cognitive_module_profile
from llmintent.heighten import (
    FocusMetrics,
    HeightenedReasoningFramework,
    HeightenedReasoningResult,
    RetraceMode,
    heighten_reasoning,
)
from llmintent.jspace import (
    IntentTrace,
    TransportMaps,
    build_intent_trace,
    classify_layer_regimes,
    fit_transport_maps,
)
from llmintent.kernels import minimize_twin_barlow, per_layer_kl_profile
from llmintent.layers import build_layer_correspondence_map, summarize_layer_bands
from llmintent.metrics import calculate_sso_score, kl_divergence, shannon_entropy
from llmintent.query import ConceptQueryResult, query_concept_in_trajectory, query_concepts_batch
from llmintent.trajectory import TrajectoryMapping, build_trajectory_mapping
from llmintent.viz import VisualizationSuite
from llmintent.benchmark import (
    AblationCondition,
    BenchmarkRunConfig,
    HellaSwagBenchmarkRunner,
    RetraceStore,
    list_slms,
    parse_conditions,
    prepare_slm_comparison,
)
from llmintent.retracement import (
    RetracementConfig,
    RetracementMode,
    RetracementTransformer,
    run_retracement_ablation,
)

__all__ = [
    "AnalysisReport",
    "CognitiveModuleProfile",
    "AblationCondition",
    "BenchmarkRunConfig",
    "CompactionAnalyzer",
    "ConceptQueryResult",
    "FocusMetrics",
    "HellaSwagBenchmarkRunner",
    "HeightenedReasoningFramework",
    "HeightenedReasoningResult",
    "IntentTrace",
    "LLMIntentAnalyzer",
    "RetraceStore",
    "RetraceMode",
    "RetracementConfig",
    "RetracementMode",
    "RetracementTransformer",
    "TrajectoryMapping",
    "TransportMaps",
    "VisualizationSuite",
    "activation_summary",
    "build_cognitive_module_profile",
    "build_intent_trace",
    "build_layer_correspondence_map",
    "build_trajectory_mapping",
    "calculate_sso_score",
    "classify_layer_regimes",
    "fit_transport_maps",
    "heighten_reasoning",
    "identify_activation_layers",
    "kl_divergence",
    "minimize_twin_barlow",
    "list_slms",
    "parse_conditions",
    "prepare_slm_comparison",
    "per_layer_kl_profile",
    "query_concept_in_trajectory",
    "query_concepts_batch",
    "run_retracement_ablation",
    "shannon_entropy",
    "summarize_layer_bands",
]

__version__ = "0.8.0"
