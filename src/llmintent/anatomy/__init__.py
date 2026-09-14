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
from llmintent.anatomy.svd_map import (
    SVDAnatomy,
    map_activations,
    map_synthetic,
    map_weights,
    match_text_to_region,
)
from llmintent.anatomy.thoughts import LatentThoughtReport, inspect_latent_thoughts
from llmintent.anatomy.trace import PromptTrace, RegionTrace, trace_prompt
from llmintent.anatomy.trajectory import AnatomyTrajectory, trajectory

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
    "ablate_linear",
    "ablate_model",
    "abstract_layer",
    "compile_regions",
    "connectome_iv",
    "default_atlas",
    "draft_anatomy_report",
    "INTENTS",
    "MisAlignFlag",
    "inspect_latent_thoughts",
    "intent_ids",
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
