"""
LLMIntent Live — real-time focused reasoning suite.

Load Phi-3, Qwen 0.5B, or other SLMs once; analyze, heighten, and generate
with Retracement Transformer and activation steering in interactive latency.
"""

from llmintent.live.api import create_app, serve
from llmintent.live.pipeline import LiveIntentPipeline
from llmintent.live.registry import (
    LIVE_MODELS,
    default_live_model,
    get_live_model,
    list_live_models,
)
from llmintent.live.session import LiveSession, LiveSessionConfig
from llmintent.live.types import (
    LiveAnalyzeResult,
    LiveGenerateResult,
    LiveHeightenResult,
    LiveModelSpec,
)

__all__ = [
    "LIVE_MODELS",
    "LiveAnalyzeResult",
    "LiveGenerateResult",
    "LiveHeightenResult",
    "LiveIntentPipeline",
    "LiveModelSpec",
    "LiveSession",
    "LiveSessionConfig",
    "create_app",
    "default_live_model",
    "get_live_model",
    "list_live_models",
    "serve",
]
