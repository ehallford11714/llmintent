"""Real-time intent pipeline — fast analyze, heighten, and generate."""

from __future__ import annotations

import time

import pandas as pd
import torch

from llmintent.activation import identify_activation_layers
from llmintent.heighten.focus import compute_focus_metrics
from llmintent.heighten.intervention import extract_reasoning_focus_vector, steering_hooks
from llmintent.heighten.retrace import build_focused_prompt, build_retrace_prompt
from llmintent.heighten.types import RetraceMode
from llmintent.live.activation_stream import (
    iter_generate_with_layer_activations,
    snapshot_layer_activations,
)
from llmintent.live.generate import format_prompt, generate_completion, next_token_topk
from llmintent.live.session import LiveSession, LiveSessionConfig
from llmintent.live.types import LiveAnalyzeResult, LiveGenerateResult, LiveHeightenResult
from llmintent.trajectory import build_trajectory_mapping


def _infer_concepts(prompt: str) -> list[str]:
    words = [w.strip(".,!?;:") for w in prompt.split() if len(w) > 3]
    return list(dict.fromkeys(words))[-5:] or ["reasoning", "answer"]


class LiveIntentPipeline:
    """
    Real-time orchestrator for loaded SLMs (Phi-3, Qwen 0.5B, etc.).

    Designed for interactive apps — one hot model, lightweight analysis paths,
    optional retracement and steering on generate.
    """

    def __init__(self, config: LiveSessionConfig | None = None) -> None:
        self.session = LiveSession(config)

    @property
    def model_key(self) -> str:
        return self.session.model_key

    def load(self, model_key: str | None = None) -> None:
        self.session.load(model_key)

    def unload(self) -> None:
        self.session.unload()

    def analyze(
        self,
        prompt: str,
        *,
        concepts: list[str] | None = None,
        include_focus: bool = True,
    ) -> LiveAnalyzeResult:
        """Fast activation + optional lightweight focus diagnosis."""
        t0 = time.perf_counter()
        bundle = self.session.bundle
        concepts = concepts or _infer_concepts(prompt)

        activation_layers = identify_activation_layers(bundle, prompt)
        pivots = {k: int(v) for k, v in activation_layers.items()}

        focus_score = None
        needs_retrace = None
        recommended: list[int] = []

        if include_focus:
            mapping = build_trajectory_mapping(
                bundle,
                prompt,
                include_cognitive=False,
                include_concepts=False,
            )
            metrics = compute_focus_metrics(mapping)
            focus_score = metrics.focus_score
            needs_retrace = metrics.needs_retrace
            recommended = metrics.recommended_focus_layers
            pivots = mapping.pivots or pivots

        elapsed = (time.perf_counter() - t0) * 1000
        return LiveAnalyzeResult(
            model_key=self.session.model_key,
            prompt=prompt,
            activation_layers=activation_layers,
            focus_score=focus_score,
            needs_retrace=needs_retrace,
            pivots=pivots,
            recommended_focus_layers=recommended,
            latency_ms=elapsed,
        )

    def heighten(
        self,
        prompt: str,
        *,
        anchor_prompt: str | None = None,
        concepts: list[str] | None = None,
        mode: str = "explicit_retrace",
        steer: bool = False,
    ) -> LiveHeightenResult:
        """Build retrace scaffold and optionally apply one-shot steering probe."""
        t0 = time.perf_counter()
        bundle = self.session.bundle
        concepts = concepts or _infer_concepts(prompt)
        anchor = anchor_prompt or prompt

        retrace_mode = RetraceMode(mode)
        retrace_prompt = build_retrace_prompt(anchor, mode=retrace_mode, concepts=concepts)

        mapping_before = build_trajectory_mapping(
            bundle, prompt, twin_b=anchor, include_cognitive=False, include_concepts=False
        )
        focus_before = compute_focus_metrics(mapping_before).focus_score

        mapping_after = build_trajectory_mapping(
            bundle, prompt, twin_b=retrace_prompt, include_cognitive=False, include_concepts=False
        )
        focus_after = compute_focus_metrics(mapping_after).focus_score

        top_shift: dict[str, float] = {}
        if steer:
            vec = extract_reasoning_focus_vector(bundle, anchor, retrace_prompt)
            layers = mapping_before.pivots.values() or [bundle.num_layers // 2]
            layer_list = sorted(set(int(x) for x in layers))[:3]

            enc = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
            with steering_hooks(bundle, layer_list, vec, self.session.config.steering_coefficient):
                out = bundle.model(enc.input_ids)
            logits_steered = out.logits if hasattr(out, "logits") else out[0]

            out_base = bundle.model(enc.input_ids)
            logits_base = out_base.logits if hasattr(out_base, "logits") else out_base[0]

            delta = (logits_steered[0, -1] - logits_base[0, -1]).float()
            top_idx = torch.topk(delta.abs(), min(5, delta.numel())).indices.tolist()
            for idx in top_idx:
                tok = bundle.tokenizer.decode([idx])
                top_shift[tok] = float(delta[idx].item())

        elapsed = (time.perf_counter() - t0) * 1000
        return LiveHeightenResult(
            model_key=self.session.model_key,
            prompt=prompt,
            retrace_prompt=retrace_prompt,
            focus_before=focus_before,
            focus_after=focus_after,
            focus_gain=(focus_after - focus_before) if focus_after is not None and focus_before is not None else None,
            steering_applied=steer,
            top_logits_shift=top_shift,
            latency_ms=elapsed,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        retracement_mode: str | None = None,
        steer: bool = False,
        anchor_prompt: str | None = None,
    ) -> LiveGenerateResult:
        t0 = time.perf_counter()
        completion, steered = generate_completion(
            self.session,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            retracement_mode=retracement_mode,
            steer=steer,
            anchor_prompt=anchor_prompt,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return LiveGenerateResult(
            model_key=self.session.model_key,
            prompt=prompt,
            completion=completion,
            retracement_mode=retracement_mode or self.session.config.retracement_mode,
            steered=steered,
            latency_ms=elapsed,
        )

    def probe_next_tokens(
        self,
        prompt: str,
        *,
        k: int = 5,
        retracement_mode: str | None = None,
    ) -> list[tuple[str, float]]:
        return next_token_topk(
            self.session,
            prompt,
            k=k,
            retracement_mode=retracement_mode,
        )

    def snapshot_prompt_layers(self, prompt: str) -> pd.DataFrame:
        """Per-layer activation metrics for the current prompt (single forward pass)."""
        bundle = self.session.bundle
        text = format_prompt(bundle, prompt, use_chat=self.session.spec.chat_template)
        enc = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
        return snapshot_layer_activations(self.session, enc.input_ids)

    def stream_layer_activations(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 16,
        temperature: float = 0.7,
        retracement_mode: str | None = None,
    ):
        """Yield per-token layer activation snapshots during generation."""
        yield from iter_generate_with_layer_activations(
            self.session,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            retracement_mode=retracement_mode,
        )

    def focused_prefix(self, prompt: str, *, concepts: list[str] | None = None) -> str:
        return build_focused_prompt(prompt, concepts=concepts or _infer_concepts(prompt))
