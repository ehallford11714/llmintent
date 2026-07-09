"""Dedicated model suite: Qwen, Mistral, MiniMax, GLM (+ legacy SLMs)."""

from llmintent.suite.loader import load_suite_model, soft_pipeline
from llmintent.suite.registry import (
    FAMILIES,
    SIZES,
    ModelSpec,
    get_model_spec,
    list_families,
    list_models,
    list_sizes,
)
from llmintent.suite.resolve import (
    resolve_from_env,
    resolve_model_id,
    resolve_model_spec,
)

__all__ = [
    "FAMILIES",
    "SIZES",
    "ModelSpec",
    "get_model_spec",
    "list_families",
    "list_models",
    "list_sizes",
    "load_suite_model",
    "resolve_from_env",
    "resolve_model_id",
    "resolve_model_spec",
    "soft_pipeline",
]
