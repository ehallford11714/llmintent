"""Ablation study for Retracement Transformer modes."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from llmintent.models import load_model_bundle
from llmintent.retracement.config import RetracementConfig, RetracementMode
from llmintent.retracement.perplexity import PerplexityResult, compute_perplexity, load_eval_texts
from llmintent.retracement.transformer import RetracementTransformer


RETRACEMENT_ABLATION_MODES: tuple[RetracementMode, ...] = (
    RetracementMode.BASELINE,
    RetracementMode.FOCUS_GATE,
    RetracementMode.RETRACE_STEER,
    RetracementMode.DUAL_PASS,
    RetracementMode.WORKSPACE_LOOP,
    RetracementMode.EXTREME,
)

FAST_RETRACEMENT_ABLATION: tuple[RetracementMode, ...] = (
    RetracementMode.BASELINE,
    RetracementMode.FOCUS_GATE,
    RetracementMode.DUAL_PASS,
    RetracementMode.EXTREME,
)


@dataclass
class RetracementAblationConfig:
    models: list[str] = field(default_factory=lambda: ["gpt2", "distilgpt2"])
    modes: list[RetracementMode] = field(default_factory=lambda: list(FAST_RETRACEMENT_ABLATION))
    text_limit: int = 24
    max_length: int = 128
    focus_coefficient: float = 0.35
    retrace_coefficient: float = 0.5


@dataclass
class RetracementAblationResult:
    model_name: str
    mode: str
    perplexity: float
    avg_nll: float
    delta_ppl_vs_baseline: float | None
    delta_nll_vs_baseline: float | None
    num_tokens: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "perplexity": self.perplexity,
            "avg_nll": self.avg_nll,
            "delta_ppl_vs_baseline": self.delta_ppl_vs_baseline,
            "delta_nll_vs_baseline": self.delta_nll_vs_baseline,
            "num_tokens": self.num_tokens,
        }


class RetracementAblationRunner:
    """Run perplexity ablation across Retracement Transformer modes."""

    def __init__(self, config: RetracementAblationConfig | None = None) -> None:
        self.config = config or RetracementAblationConfig()

    def run_model(self, model_name: str) -> list[RetracementAblationResult]:
        texts = load_eval_texts(limit=self.config.text_limit)
        bundle = load_model_bundle(model_name)
        results: list[PerplexityResult] = []

        try:
            for mode in self.config.modes:
                cfg = RetracementConfig(
                    mode=mode,
                    focus_coefficient=self.config.focus_coefficient,
                    retrace_coefficient=self.config.retrace_coefficient,
                )
                rt = RetracementTransformer(bundle, cfg)
                results.append(
                    compute_perplexity(rt, texts, max_length=self.config.max_length)
                )
        finally:
            del bundle.model

        baseline = next((r for r in results if r.mode == RetracementMode.BASELINE.value), None)
        out: list[RetracementAblationResult] = []
        for r in results:
            delta_ppl = None
            delta_nll = None
            if baseline and r.mode != baseline.mode:
                delta_ppl = r.perplexity - baseline.perplexity
                delta_nll = r.avg_nll - baseline.avg_nll
            out.append(
                RetracementAblationResult(
                    model_name=r.model_name,
                    mode=r.mode,
                    perplexity=r.perplexity,
                    avg_nll=r.avg_nll,
                    delta_ppl_vs_baseline=delta_ppl,
                    delta_nll_vs_baseline=delta_nll,
                    num_tokens=r.num_tokens,
                )
            )
        return out

    def run_all(self) -> pd.DataFrame:
        rows: list[dict] = []
        for model in self.config.models:
            for res in self.run_model(model):
                rows.append(res.to_dict())
        return pd.DataFrame(rows)


def run_retracement_ablation(
    models: list[str] | None = None,
    *,
    fast: bool = True,
    text_limit: int = 24,
) -> pd.DataFrame:
    """Convenience entry point for notebooks and CLI."""
    modes = list(FAST_RETRACEMENT_ABLATION if fast else RETRACEMENT_ABLATION_MODES)
    cfg = RetracementAblationConfig(
        models=models or ["gpt2", "distilgpt2"],
        modes=modes,
        text_limit=text_limit,
    )
    return RetracementAblationRunner(cfg).run_all()
