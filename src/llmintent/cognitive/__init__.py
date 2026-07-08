"""Cognitive module kernels: identity, reasoning, meta-reasoning, ideation."""

from llmintent.cognitive.orchestrator import (
    build_cognitive_module_profile,
    enrich_layer_map_with_cognitive_modules,
)
from llmintent.cognitive.types import COGNITIVE_MODULES, CognitiveKernel, CognitiveModuleProfile

__all__ = [
    "COGNITIVE_MODULES",
    "CognitiveKernel",
    "CognitiveModuleProfile",
    "build_cognitive_module_profile",
    "enrich_layer_map_with_cognitive_modules",
]
