"""High-level facade combining notebook extraction workflows."""

from __future__ import annotations

import gc
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
import torch

from llmintent.activation import activation_summary, identify_activation_layers
from llmintent.compaction import CompactionAnalyzer
from llmintent.embeddings import EmbeddingSpace, load_glove_gensim
from llmintent.cognitive import CognitiveModuleProfile, build_cognitive_module_profile
from llmintent.jspace.trace import IntentTrace, build_intent_trace
from llmintent.jspace.transport import TransportMaps, fit_transport_maps
from llmintent.layers import build_layer_correspondence_map, summarize_layer_bands
from llmintent.models import ModelBundle, get_transformer_layers, load_model_bundle
from llmintent.morphemes import MorphemeExtractor
from llmintent.poles import build_glove_poles, build_numerical_pole
from llmintent.query import ConceptQueryResult, query_concept_in_trajectory, query_concepts_batch
from llmintent.trajectory import TrajectoryMapping, build_trajectory_mapping
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
    activation_layers: dict[str, int] = field(default_factory=dict)
    layer_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    intent_trace: IntentTrace | None = None
    cognitive_profile: CognitiveModuleProfile | None = None


class LLMIntentAnalyzer:
    """
    Unified semantic extraction analyzer.

    Combines notebook metrics (steering, compaction, morpheme wells) with
    Anthropic J-space layer thoughts (logit/J-lens decode, activation pivots).
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
        fit_jspace_transport: bool = False,
        transport_prompts: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.pivot_layer = pivot_layer
        self.bundle = load_model_bundle(model_name, device=device)
        self.extractor = MorphemeExtractor(morpheme_backend)  # type: ignore[arg-type]
        self.embedding_space: EmbeddingSpace | None = None
        self.morpheme_freq: Counter[str] = Counter()
        self.transport: TransportMaps | None = None

        if load_glove:
            self.embedding_space = load_glove_gensim(embedding_name)
            sample_words = self.embedding_space.vocab[:5000]
            self.morpheme_freq = Counter(self.extractor.extract(sample_words))

        self._numerical_pole: torch.Tensor | None = None
        self._glove_poles = build_glove_poles(self.embedding_space) if self.embedding_space else None

        if fit_jspace_transport:
            prompts = transport_prompts or [
                "The quick brown fox jumps over the lazy dog.",
                "Two plus two equals four.",
                "Question: What is 12 times 2? Answer:",
            ]
            self.transport = fit_transport_maps(self.bundle, prompts)

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
        include_jspace: bool = True,
        track_tokens: list[str] | None = None,
        twin_b: str | None = None,
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

        if include_jspace:
            report.intent_trace = build_intent_trace(
                self.bundle,
                prompt,
                transport=self.transport,
                track_tokens=track_tokens,
            )
            report.activation_layers = report.intent_trace.activation_layers
            if twin_b:
                report.cognitive_profile = build_cognitive_module_profile(
                    self.bundle,
                    prompt,
                    twin_b,
                    transport=self.transport,
                )
                report.layer_map = build_layer_correspondence_map(
                    self.bundle,
                    prompt,
                    transport=self.transport,
                    cognitive_profile=report.cognitive_profile,
                )
            else:
                report.layer_map = build_layer_correspondence_map(
                    self.bundle,
                    prompt,
                    transport=self.transport,
                )
            report.inference_pivot = report.activation_layers.get("inference_pivot")

        if include_block_semantics and self.embedding_space:
            report.block_semantics = self.extract_block_semantics()

        if include_compaction and self.embedding_space:
            comp = CompactionAnalyzer(self.model_name, self.embedding_space)
            report.compaction = comp.analyze_compaction()
            if report.inference_pivot is None:
                report.inference_pivot = comp.find_inference_pivot(report.compaction)
            comp.cleanup()

        return report

    def identify_activation(self, prompt: str) -> dict[str, int]:
        """Return layer indices for inference pivot, workspace peak, motor onset."""
        return identify_activation_layers(self.bundle, prompt)

    def cognitive_modules(
        self,
        twin_a: str,
        twin_b: str,
    ) -> CognitiveModuleProfile:
        """Identify identity, reasoning, meta-reasoning, ideation kernels."""
        return build_cognitive_module_profile(
            self.bundle,
            twin_a,
            twin_b,
            transport=self.transport,
        )

    def layer_correspondence(
        self,
        prompt: str,
        *,
        twin_b: str | None = None,
    ) -> pd.DataFrame:
        """Map each transformer layer to regime, role, and top verbal intent."""
        return build_layer_correspondence_map(
            self.bundle,
            prompt,
            transport=self.transport,
            twin_b=twin_b,
        )

    def intent_trace(
        self,
        prompt: str,
        *,
        track_tokens: list[str] | None = None,
    ) -> IntentTrace:
        """Build full J-space intent trace (layer thoughts)."""
        return build_intent_trace(
            self.bundle,
            prompt,
            transport=self.transport,
            track_tokens=track_tokens,
        )

    def layer_band_summary(self, prompt: str) -> dict:
        return summarize_layer_bands(self.bundle, prompt, transport=self.transport)

    def fit_transport(self, prompts: list[str]) -> TransportMaps:
        self.transport = fit_transport_maps(self.bundle, prompts)
        return self.transport

    def query_concept(
        self,
        concept: str,
        prompt: str,
        *,
        twin_b: str | None = None,
        top_k_layers: int = 5,
    ) -> ConceptQueryResult:
        """
        Query a semantic concept against the activation trajectory.

        Uses KL + twin Barlow feature space with KNN retrieval to identify
        which layers activate for the given concept text.
        """
        cognitive = None
        if twin_b:
            cognitive = build_cognitive_module_profile(
                self.bundle,
                prompt,
                twin_b,
                transport=self.transport,
            )
        return query_concept_in_trajectory(
            self.bundle,
            concept,
            prompt,
            twin_b=twin_b,
            top_k_layers=top_k_layers,
            cognitive_profile=cognitive,
        )

    def query_concepts(
        self,
        concepts: list[str],
        prompt: str,
        *,
        twin_b: str | None = None,
    ) -> dict[str, ConceptQueryResult]:
        return query_concepts_batch(
            self.bundle,
            concepts,
            prompt,
            twin_b=twin_b,
        )

    def trajectory_map(
        self,
        prompt: str,
        *,
        twin_b: str | None = None,
        concepts: list[str] | None = None,
    ) -> TrajectoryMapping:
        """Build unified activation trajectory mapping across all layers."""
        return build_trajectory_mapping(
            self.bundle,
            prompt,
            twin_b=twin_b,
            transport=self.transport,
            concepts=concepts,
        )

    def compare_prompts(
        self,
        prompts: dict[str, str],
    ) -> pd.DataFrame:
        """Multi-prompt intensity sweep (Direct vs CoT, ablation levels, etc.)."""
        return run_intensity_sweep(self.bundle, prompts, self.numerical_pole)

    def stress_test(self, simple: str, complex: str) -> pd.DataFrame:
        return run_stress_test(self.bundle, simple, complex)

    def activation_profile(self, prompt: str) -> pd.DataFrame:
        return activation_summary(self.bundle, prompt)

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
        canonical = 18
        canonical_depth = 48
        scaled = int(round(canonical / canonical_depth * self.bundle.num_layers))
        return max(1, min(scaled, self.bundle.num_layers - 1))

    def cleanup(self) -> None:
        del self.bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
