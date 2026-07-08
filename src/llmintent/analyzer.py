"""High-level facade combining notebook extraction workflows."""

from __future__ import annotations

import gc
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
import torch

from llmintent.compaction import CompactionAnalyzer
from llmintent.embeddings import EmbeddingSpace, load_glove_gensim
from llmintent.models import ModelBundle, get_transformer_layers, load_model_bundle
from llmintent.morphemes import MorphemeExtractor
from llmintent.poles import build_glove_poles, build_numerical_pole
from llmintent.steering import (
    analyze_steering_intensity,
    calculate_pivot_entropy,
    compare_cot_intensity,
    get_entropy_trajectory,
    run_intensity_sweep,
    run_stress_test,
)
from llmintent.weight_semantics import get_block_expertise, get_block_semantics


@dataclass
class AnalysisReport:
    prompt: str
    model_name: str
    intensity_sweep: pd.DataFrame = field(default_factory=pd.DataFrame)
    entropy_trajectory: pd.DataFrame = field(default_factory=pd.DataFrame)
    pivot_entropy: dict[str, float] = field(default_factory=dict)
    cot_comparison: dict[str, float] = field(default_factory=dict)
    block_semantics: dict[int, dict[str, list[str]]] = field(default_factory=dict)
    compaction: pd.DataFrame = field(default_factory=pd.DataFrame)
    inference_pivot: int | None = None


class LLMIntentAnalyzer:
    """
    Unified semantic extraction analyzer.

    Mirrors the SemanticExtractionLLms notebook pipeline:
    morpheme wells → block semantics → steering intensity → compaction/SSO.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        morpheme_backend: str = "lemma",
        embedding_name: str = "glove-wiki-gigaword-100",
        load_glove: bool = True,
        pivot_layer: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.pivot_layer = pivot_layer
        self.bundle = load_model_bundle(model_name, device=device)
        self.extractor = MorphemeExtractor(morpheme_backend)  # type: ignore[arg-type]
        self.embedding_space: EmbeddingSpace | None = None
        self.morpheme_freq: Counter[str] = Counter()

        if load_glove:
            self.embedding_space = load_glove_gensim(embedding_name)
            sample_words = self.embedding_space.vocab[:5000]
            self.morpheme_freq = Counter(self.extractor.extract(sample_words))

        self._numerical_pole: torch.Tensor | None = None
        self._glove_poles = build_glove_poles(self.embedding_space) if self.embedding_space else None

    @property
    def numerical_pole(self) -> torch.Tensor:
        if self._numerical_pole is None:
            self._numerical_pole = build_numerical_pole(self.bundle)
        return self._numerical_pole

    def analyze_prompt(
        self,
        prompt: str,
        *,
        cot_prompt: str | None = None,
        include_compaction: bool = False,
        include_block_semantics: bool = False,
    ) -> AnalysisReport:
        report = AnalysisReport(prompt=prompt, model_name=self.model_name)
        pole = self.numerical_pole

        report.intensity_sweep = analyze_steering_intensity(self.bundle, prompt, pole)
        report.entropy_trajectory = get_entropy_trajectory(self.bundle, prompt)

        pivot = self.pivot_layer or self._default_pivot()
        if cot_prompt:
            report.cot_comparison = compare_cot_intensity(
                self.bundle,
                prompt,
                cot_prompt,
                pole,
                pivot_layer=pivot,
            )
            report.pivot_entropy = calculate_pivot_entropy(
                self.bundle,
                prompt,
                cot_prompt,
                pivot_layer=pivot,
            )

        if include_block_semantics and self.embedding_space:
            report.block_semantics = self.extract_block_semantics()

        if include_compaction and self.embedding_space:
            comp = CompactionAnalyzer(self.model_name, self.embedding_space)
            report.compaction = comp.analyze_compaction()
            report.inference_pivot = comp.find_inference_pivot(report.compaction)
            comp.cleanup()

        return report

    def compare_prompts(
        self,
        prompts: dict[str, str],
    ) -> pd.DataFrame:
        """Multi-prompt intensity sweep (Direct vs CoT, ablation levels, etc.)."""
        return run_intensity_sweep(self.bundle, prompts, self.numerical_pole)

    def stress_test(self, simple: str, complex: str) -> pd.DataFrame:
        return run_stress_test(self.bundle, simple, complex)

    def extract_block_semantics(self, max_layers: int | None = None) -> dict[int, dict[str, list[str]]]:
        if not self.embedding_space:
            raise RuntimeError("GloVe embeddings required for block semantics")
        layers = get_transformer_layers(self.bundle.model)
        limit = max_layers or len(layers)
        out: dict[int, dict[str, list[str]]] = {}
        for idx, layer in enumerate(layers[:limit]):
            try:
                out[idx] = get_block_semantics(
                    layer,
                    self.embedding_space,
                    self.morpheme_freq,
                    extractor=self.extractor,
                )
            except AttributeError:
                continue
        return out

    def block_expertise_report(self, max_layers: int | None = None) -> pd.DataFrame:
        semantics = self.extract_block_semantics(max_layers=max_layers)
        rows: list[dict[str, str | int]] = []
        for layer_idx, block in semantics.items():
            expertise = get_block_expertise(block)
            for component, units in block.items():
                rows.append(
                    {
                        "layer": layer_idx,
                        "expertise": expertise,
                        "component": component,
                        "top_units": ", ".join(units[:5]),
                    }
                )
        return pd.DataFrame(rows)

    def _default_pivot(self) -> int:
        # Notebook default for GPT-2 XL; scale proportionally for other depths
        canonical = 18
        canonical_depth = 48
        scaled = int(round(canonical / canonical_depth * self.bundle.num_layers))
        return max(1, min(scaled, self.bundle.num_layers - 1))

    def cleanup(self) -> None:
        del self.bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
