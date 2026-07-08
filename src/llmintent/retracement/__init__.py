"""Retracement Transformer — focused-reasoning architecture wrapper."""

from llmintent.retracement.ablation import (
    FAST_RETRACEMENT_ABLATION,
    RETRACEMENT_ABLATION_MODES,
    RetracementAblationConfig,
    RetracementAblationResult,
    RetracementAblationRunner,
    run_retracement_ablation,
)
from llmintent.retracement.blocks import FocusGate, RetraceMerge
from llmintent.retracement.config import RetracementConfig, RetracementMode
from llmintent.retracement.perplexity import PerplexityResult, compute_perplexity, evaluate_perplexity, load_eval_texts
from llmintent.retracement.transformer import RetracementState, RetracementTransformer

__all__ = [
    "FAST_RETRACEMENT_ABLATION",
    "FocusGate",
    "PerplexityResult",
    "RETRACEMENT_ABLATION_MODES",
    "RetraceMerge",
    "RetracementAblationConfig",
    "RetracementAblationResult",
    "RetracementAblationRunner",
    "RetracementConfig",
    "RetracementMode",
    "RetracementState",
    "RetracementTransformer",
    "compute_perplexity",
    "evaluate_perplexity",
    "load_eval_texts",
    "run_retracement_ablation",
]
