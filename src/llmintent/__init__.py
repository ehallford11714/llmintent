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
from llmintent.benchmark import (
    AblationCondition,
    BenchmarkRunConfig,
    HellaSwagBenchmarkRunner,
    RetraceStore,
    list_slms,
    parse_conditions,
    prepare_slm_comparison,
)
from llmintent.live import (
    LiveIntentPipeline,
    LiveSessionConfig,
    list_live_models,
)
from llmintent.retracement import (
    RetracementConfig,
    RetracementMode,
    RetracementTransformer,
    run_retracement_ablation,
)
from llmintent.suite import (
    get_model_spec,
    list_models,
    load_suite_model,
    resolve_model_id,
)
from llmintent.anatomy import (
    Anatomy,
    AnatomyReport,
    compile_regions,
    draft_anatomy_report,
    map_anatomy,
    trajectory,
    trace_prompt,
    what_region_does,
)
from llmintent.entity import PersistenceReport, persist_entities
from llmintent.persistbind import BindReport, BoundState, PersistenceBinder, persistence_bind
from llmintent.instream import bind_positions, bind_within_stream, forward_within
from llmintent.memslot import attach_memory_slot
from llmintent.worldfile import WorldFile, apply_world
from llmintent.predictbind import bayes_inform, greedy_generate, predict_from_bind
from llmintent.spike import SpikeConfig, SpikePotentialBank, node_adjoint

__all__ = [
    "AnalysisReport",
    "AnatomyReport",
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
    "LiveIntentPipeline",
    "LiveSessionConfig",
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
    "compile_regions",
    "draft_anatomy_report",
    "fit_transport_maps",
    "get_model_spec",
    "heighten_reasoning",
    "identify_activation_layers",
    "kl_divergence",
    "list_models",
    "load_suite_model",
    "minimize_twin_barlow",
    "list_live_models",
    "list_slms",
    "map_anatomy",
    "Anatomy",
    "parse_conditions",
    "persist_entities",
    "PersistenceReport",
    "persistence_bind",
    "PersistenceBinder",
    "BindReport",
    "BoundState",
    "bind_positions",
    "bind_within_stream",
    "forward_within",
    "attach_memory_slot",
    "WorldFile",
    "apply_world",
    "predict_from_bind",
    "bayes_inform",
    "greedy_generate",
    "SpikeConfig",
    "SpikePotentialBank",
    "node_adjoint",
    "trace_prompt",
    "trajectory",
    "prepare_slm_comparison",
    "per_layer_kl_profile",
    "query_concept_in_trajectory",
    "query_concepts_batch",
    "resolve_model_id",
    "run_retracement_ablation",
    "shannon_entropy",
    "summarize_layer_bands",
    "what_region_does",
]


def __getattr__(name: str):
    if name == "VisualizationSuite":
        from llmintent.viz import VisualizationSuite as _VisualizationSuite

        return _VisualizationSuite
    raise AttributeError(f"module 'llmintent' has no attribute {name!r}")


__version__ = "1.8.0"
