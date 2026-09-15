"""Fly-connectome-guided anatomy of a transformer LLM.

Compile English onto closed regions (what each handles), use literature-core
wiring as the IV prior (how regions integrate), SVD-map residuals/FFN onto
those regions, and ablate region A vs B to show the output changes.
"""

from __future__ import annotations

from llmintent.anatomy.ablation import (
    AblationResult,
    ablate_linear,
    ablate_model,
    plant_and_ablate,
)
from llmintent.anatomy.atlas import (
    REGIONS,
    REGION_BY_ID,
    Atlas,
    Region,
    abstract_layer,
    layers_for_region,
    region_ids,
    what_region_does,
)
from llmintent.anatomy.compile import RegionPlan, compile_regions
from llmintent.anatomy.connectome import (
    RegionConnectome,
    default_atlas,
    literature_region_connectome,
)
from llmintent.anatomy.iv_engine import AnatomyIVResult, connectome_iv, iv_from_text
from llmintent.anatomy.report import AnatomyReport, RegionCard, map_anatomy
from llmintent.anatomy.guide import GuideDraft, draft_anatomy_report
from llmintent.anatomy.intents import INTENTS, intent_ids
from llmintent.anatomy.misalign import MisAlignFlag, scan_negative_intent
from llmintent.anatomy.model_map import Anatomy, AnatomyGraph
from llmintent.anatomy.experiment import run_atlas_experiment
from llmintent.anatomy.fly_assay import FlyFunctionalAssay, run_looming_giant_fibre_assay
from llmintent.anatomy.region_test import test_analogous_region, validate_analogous_region_test
from llmintent.anatomy.svd_map import (
    SVDAnatomy,
    map_activations,
    map_synthetic,
    map_weights,
    match_text_to_region,
)
from llmintent.anatomy.intent_track import (
    IntentTrack,
    track_from_residuals,
    track_latent_intent,
    track_prompt_compile,
)
from llmintent.anatomy.thoughts import LatentThoughtReport, inspect_latent_thoughts
from llmintent.anatomy.trace import PromptTrace, RegionTrace, trace_prompt
from llmintent.anatomy.trajectory import AnatomyTrajectory, trajectory
from llmintent.entity import (
    EntitySlot,
    PersistenceReport,
    detect_entity_spans,
    persist_entities,
    write_slots,
)
from llmintent.persistbind import BindReport, BoundState, PersistenceBinder, persistence_bind
from llmintent.instream import bind_positions, bind_within_stream, forward_within
from llmintent.spike import SpikeConfig, SpikePotentialBank

__all__ = [
    "AblationResult",
    "Anatomy",
    "AnatomyGraph",
    "AnatomyIVResult",
    "AnatomyReport",
    "AnatomyTrajectory",
    "Atlas",
    "GuideDraft",
    "LatentThoughtReport",
    "PromptTrace",
    "REGIONS",
    "REGION_BY_ID",
    "Region",
    "RegionCard",
    "RegionConnectome",
    "RegionPlan",
    "RegionTrace",
    "SVDAnatomy",
    "FlyFunctionalAssay",
    "run_atlas_experiment",
    "run_looming_giant_fibre_assay",
    "test_analogous_region",
    "validate_analogous_region_test",
    "ablate_linear",
    "ablate_model",
    "abstract_layer",
    "compile_regions",
    "detect_entity_spans",
    "EntitySlot",
    "persist_entities",
    "PersistenceReport",
    "persistence_bind",
    "PersistenceBinder",
    "BindReport",
    "BoundState",
    "bind_positions",
    "bind_within_stream",
    "forward_within",
    "SpikeConfig",
    "SpikePotentialBank",
    "write_slots",
    "connectome_iv",
    "default_atlas",
    "draft_anatomy_report",
    "INTENTS",
    "MisAlignFlag",
    "inspect_latent_thoughts",
    "IntentTrack",
    "intent_ids",
    "track_from_residuals",
    "track_latent_intent",
    "track_prompt_compile",
    "iv_from_text",
    "layers_for_region",
    "literature_region_connectome",
    "map_activations",
    "map_anatomy",
    "map_synthetic",
    "map_weights",
    "match_text_to_region",
    "plant_and_ablate",
    "region_ids",
    "scan_negative_intent",
    "trace_prompt",
    "trajectory",
    "what_region_does",
]
