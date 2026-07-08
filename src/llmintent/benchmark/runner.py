"""HellaSwag benchmark runner with focused/extreme retrace ablations on SLMs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from llmintent.benchmark.ablation import AblationCondition
from llmintent.benchmark.hellaswag import (
    HellaSwagExample,
    load_hellaswag,
    load_hellaswag_fallback,
    score_hellaswag_example,
)
from llmintent.benchmark.retrace_store import RetraceStore
from llmintent.benchmark.slm_registry import SLMConfig, get_slm
from llmintent.heighten.extreme import (
    ExtremeRetraceMode,
    build_extreme_retrace_chain,
    heighten_until_focused,
)
from llmintent.heighten.focus import compute_focus_metrics
from llmintent.heighten.intervention import extract_reasoning_focus_vector, steering_hooks
from llmintent.heighten.retrace import build_focused_prompt, build_retrace_prompt
from llmintent.heighten.types import RetraceMode
from llmintent.models import ModelBundle, load_model_bundle
from llmintent.trajectory import build_trajectory_mapping


@dataclass
class BenchmarkRunConfig:
    models: list[str] = field(default_factory=lambda: ["gpt2", "distilgpt2"])
    conditions: list[AblationCondition] = field(default_factory=lambda: list(AblationCondition))
    limit: int = 50
    split: str = "validation"
    store_path: str = "llmintent_retraces/hellaswag.jsonl"
    measure_focus: bool = True
    focus_threshold: float = 0.55
    steering_coefficient: float = 0.5
    use_fallback: bool = False
    extreme_mode: str = ExtremeRetraceMode.TRIPLE.value


@dataclass
class ConditionResult:
    model_name: str
    condition: str
    accuracy: float
    correct: int
    total: int
    mean_focus_baseline: float | None
    mean_focus_after: float | None
    mean_focus_gain: float | None


def _infer_concepts(context: str) -> list[str]:
    words = [w.strip(".,!?;:") for w in context.split() if len(w) > 3]
    return list(dict.fromkeys(words))[-5:] or ["commonsense", "context"]


def _build_prefix(
    example: HellaSwagExample,
    condition: AblationCondition,
    *,
    extreme_mode: str,
) -> tuple[str, str, list[str]]:
    """Return (scoring_prefix, retrace_prompt, retrace_chain) for a condition."""
    ctx = example.context
    concepts = _infer_concepts(ctx)
    anchor = ctx

    if condition == AblationCondition.BASELINE:
        return ctx, ctx, [ctx]

    if condition == AblationCondition.FOCUSED:
        focused = build_focused_prompt(ctx, concepts=concepts)
        return focused, focused, [focused]

    if condition == AblationCondition.RETRACE:
        rp = build_retrace_prompt(anchor, mode=RetraceMode.EXPLICIT, concepts=concepts)
        return rp, rp, [rp]

    if condition == AblationCondition.EXTREME_RETRACE:
        chain = build_extreme_retrace_chain(
            anchor,
            baseline_prompt=ctx,
            concepts=concepts,
            mode=extreme_mode,
        )
        return chain.combined_prompt, chain.combined_prompt, chain.passes

    if condition in (AblationCondition.RETRACE_STEER, AblationCondition.EXTREME_STEER):
        if condition == AblationCondition.RETRACE_STEER:
            rp = build_retrace_prompt(anchor, mode=RetraceMode.CONCEPT_ANCHOR, concepts=concepts)
            return rp, rp, [rp]
        chain = build_extreme_retrace_chain(
            anchor,
            baseline_prompt=ctx,
            concepts=concepts,
            mode=ExtremeRetraceMode.CONCEPT_LOCK.value,
        )
        return chain.combined_prompt, chain.combined_prompt, chain.passes

    if condition == AblationCondition.ITERATIVE:
        rp = build_retrace_prompt(anchor, mode=RetraceMode.CORRECTION, concepts=concepts)
        return rp, rp, [rp]

    if condition == AblationCondition.EXTREME_ITERATIVE:
        chain = build_extreme_retrace_chain(
            anchor,
            baseline_prompt=ctx,
            concepts=concepts,
            mode=ExtremeRetraceMode.VERIFY_CHAIN.value,
            passes=4,
        )
        return chain.combined_prompt, chain.combined_prompt, chain.passes

    return ctx, ctx, [ctx]


class HellaSwagBenchmarkRunner:
    """Run HellaSwag with retrace ablations across SLMs; persist to RetraceStore."""

    def __init__(self, config: BenchmarkRunConfig | None = None) -> None:
        self.config = config or BenchmarkRunConfig()
        self.store = RetraceStore(self.config.store_path)

    def load_examples(self) -> list[HellaSwagExample]:
        if self.config.use_fallback:
            return load_hellaswag_fallback(limit=min(self.config.limit, 8))
        try:
            return load_hellaswag(split=self.config.split, limit=self.config.limit)
        except ImportError:
            return load_hellaswag_fallback(limit=min(self.config.limit, 8))

    def _score_with_optional_steering(
        self,
        bundle: ModelBundle,
        example: HellaSwagExample,
        condition: AblationCondition,
        prefix: str,
        retrace_prompt: str,
    ) -> tuple[int, list[float], bool]:
        steer_conditions = {
            AblationCondition.RETRACE_STEER,
            AblationCondition.EXTREME_STEER,
        }
        if condition not in steer_conditions:
            return score_hellaswag_example(bundle, example, prefix=prefix)

        vec = extract_reasoning_focus_vector(
            bundle,
            example.context,
            retrace_prompt,
        )
        mapping = build_trajectory_mapping(bundle, example.context, twin_b=retrace_prompt)
        layers = list(mapping.pivots.values())[:3] or [bundle.num_layers // 2]

        with steering_hooks(bundle, layers, vec, self.config.steering_coefficient):
            return score_hellaswag_example(bundle, example, prefix=prefix)

    def _measure_focus(
        self,
        bundle: ModelBundle,
        context: str,
        retrace_prompt: str,
        condition: AblationCondition,
    ) -> tuple[float | None, float | None]:
        if not self.config.measure_focus:
            return None, None
        try:
            baseline_map = build_trajectory_mapping(
                bundle,
                context,
                include_cognitive=False,
            )
            baseline = compute_focus_metrics(baseline_map).focus_score

            if condition in (
                AblationCondition.ITERATIVE,
                AblationCondition.EXTREME_ITERATIVE,
            ):
                extreme = condition == AblationCondition.EXTREME_ITERATIVE
                final_focus, _, _, _ = heighten_until_focused(
                    bundle,
                    context,
                    anchor_prompt=context,
                    concepts=_infer_concepts(context),
                    focus_threshold=self.config.focus_threshold,
                    extreme=extreme,
                )
                return baseline, final_focus.focus_score

            retrace_map = build_trajectory_mapping(
                bundle,
                context,
                twin_b=retrace_prompt,
                include_cognitive=True,
            )
            after = compute_focus_metrics(retrace_map).focus_score
            return baseline, after
        except Exception:
            return None, None

    def run_model(
        self,
        model_key: str,
        *,
        conditions: list[AblationCondition] | None = None,
    ) -> list[ConditionResult]:
        slm = get_slm(model_key)
        bundle = load_model_bundle(slm.hf_name)
        examples = self.load_examples()
        conditions = conditions or self.config.conditions
        results: list[ConditionResult] = []

        try:
            for condition in conditions:
                correct = 0
                focus_baselines: list[float] = []
                focus_afters: list[float] = []

                for ex in examples:
                    prefix, retrace_prompt, chain = _build_prefix(
                        ex,
                        condition,
                        extreme_mode=self.config.extreme_mode,
                    )

                    if condition in (
                        AblationCondition.ITERATIVE,
                        AblationCondition.EXTREME_ITERATIVE,
                    ):
                        _, _, chain_obj, _ = heighten_until_focused(
                            bundle,
                            ex.context,
                            anchor_prompt=ex.context,
                            concepts=_infer_concepts(ex.context),
                            focus_threshold=self.config.focus_threshold,
                            extreme=condition == AblationCondition.EXTREME_ITERATIVE,
                        )
                        if chain_obj is not None:
                            prefix = chain_obj.combined_prompt
                            retrace_prompt = chain_obj.combined_prompt
                            chain = chain_obj.passes

                    focus_b, focus_a = self._measure_focus(
                        bundle, ex.context, retrace_prompt, condition
                    )
                    if focus_b is not None:
                        focus_baselines.append(focus_b)
                    if focus_a is not None:
                        focus_afters.append(focus_a)

                    predicted, log_probs, ok = self._score_with_optional_steering(
                        bundle,
                        ex,
                        condition,
                        prefix,
                        retrace_prompt,
                    )
                    if ok:
                        correct += 1

                    self.store.save_retracement(
                        model_name=slm.hf_name,
                        benchmark="hellaswag",
                        example_id=ex.example_id,
                        condition=condition.value,
                        context=ex.context,
                        retrace_prompt=retrace_prompt,
                        retrace_chain=chain,
                        concepts=_infer_concepts(ex.context),
                        focus_baseline=focus_b,
                        focus_after=focus_a,
                        predicted=predicted,
                        label=ex.label,
                        correct=ok,
                        log_probs=log_probs,
                        steering_applied=condition
                        in (
                            AblationCondition.RETRACE_STEER,
                            AblationCondition.EXTREME_STEER,
                        ),
                        ablation=condition.value,
                        metadata={"prefix_len": len(prefix)},
                    )

                n = len(examples)
                results.append(
                    ConditionResult(
                        model_name=slm.hf_name,
                        condition=condition.value,
                        accuracy=correct / n if n else 0.0,
                        correct=correct,
                        total=n,
                        mean_focus_baseline=(
                            sum(focus_baselines) / len(focus_baselines) if focus_baselines else None
                        ),
                        mean_focus_after=(
                            sum(focus_afters) / len(focus_afters) if focus_afters else None
                        ),
                        mean_focus_gain=(
                            (sum(focus_afters) - sum(focus_baselines)) / len(focus_afters)
                            if focus_afters and focus_baselines and len(focus_afters) == len(focus_baselines)
                            else None
                        ),
                    )
                )
        finally:
            del bundle.model
        return results

    def run_all(self) -> pd.DataFrame:
        all_results: list[ConditionResult] = []
        for model_key in self.config.models:
            all_results.extend(self.run_model(model_key))
        return pd.DataFrame([r.__dict__ for r in all_results])

    def compare_from_store(self) -> pd.DataFrame:
        return self.store.summarize_accuracy()


def prepare_slm_comparison(
    models: list[str] | None = None,
    *,
    limit: int = 20,
    store_path: str = "llmintent_retraces/hellaswag.jsonl",
    fast: bool = True,
) -> pd.DataFrame:
    """
    Prepare and run SLM comparison on HellaSwag with ablation conditions.

    Convenience entry point for notebooks and CLI.
    """
    from llmintent.benchmark.ablation import FAST_ABLATION_SUITE, parse_conditions

    config = BenchmarkRunConfig(
        models=models or ["gpt2", "distilgpt2"],
        conditions=parse_conditions("fast" if fast else "default"),
        limit=limit,
        store_path=store_path,
    )
    runner = HellaSwagBenchmarkRunner(config)
    return runner.run_all()
