"""Unified visualization suite for LLMIntent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llmintent.jspace.trace import IntentTrace, build_intent_trace
from llmintent.models import ModelBundle
from llmintent.trajectory import TrajectoryMapping, build_trajectory_mapping
from llmintent.viz import animate, correlation, maps


@dataclass
class VisualizationSuite:
    """High-level visualization API for maps, correlations, and animations."""

    bundle: ModelBundle
    output_dir: str = "llmintent_viz"
    _mapping: TrajectoryMapping | None = field(default=None, repr=False)
    _trace: IntentTrace | None = field(default=None, repr=False)

    def _ensure_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def trajectory_mapping(
        self,
        prompt: str,
        *,
        twin_b: str | None = None,
        concepts: list[str] | None = None,
    ) -> TrajectoryMapping:
        self._mapping = build_trajectory_mapping(
            self.bundle,
            prompt,
            twin_b=twin_b,
            concepts=concepts,
        )
        return self._mapping

    def intent_trace(self, prompt: str, **kwargs) -> IntentTrace:
        self._trace = build_intent_trace(self.bundle, prompt, **kwargs)
        return self._trace

    # --- Maps ---

    def save_morpheme_map(
        self,
        block_semantics: dict,
        filename: str = "morpheme_map.png",
        **kwargs,
    ) -> str:
        path = str(self._ensure_dir() / filename)
        return maps.save_morpheme_map(block_semantics, path, **kwargs)

    def save_trajectory_map(
        self,
        mapping: TrajectoryMapping | None = None,
        filename: str = "trajectory_map.png",
        **kwargs,
    ) -> str:
        mapping = mapping or self._mapping
        if mapping is None:
            raise ValueError("Call trajectory_mapping() first or pass mapping=")
        path = str(self._ensure_dir() / filename)
        return maps.save_trajectory_map(mapping, path, **kwargs)

    def save_reasoning_subspace(
        self,
        prompt: str,
        *,
        mapping: TrajectoryMapping | None = None,
        filename: str = "reasoning_subspace.png",
        **kwargs,
    ) -> str:
        mapping = mapping or self._mapping
        layer_stats = mapping.layers if mapping else None
        path = str(self._ensure_dir() / filename)
        return maps.save_reasoning_subspace(
            self.bundle,
            prompt,
            path,
            layer_stats=layer_stats,
            **kwargs,
        )

    # --- Correlation matrices ---

    def save_concept_correlation(
        self,
        mapping: TrajectoryMapping | None = None,
        filename: str = "concept_correlation.png",
        **kwargs,
    ) -> str:
        mapping = mapping or self._mapping
        if mapping is None:
            raise ValueError("Call trajectory_mapping() first or pass mapping=")
        path = str(self._ensure_dir() / filename)
        return correlation.save_concept_correlation(mapping, path, **kwargs)

    def save_reasoning_correlation(
        self,
        mapping: TrajectoryMapping | None = None,
        filename: str = "reasoning_trace_correlation.png",
        **kwargs,
    ) -> str:
        mapping = mapping or self._mapping
        if mapping is None:
            raise ValueError("Call trajectory_mapping() first or pass mapping=")
        path = str(self._ensure_dir() / filename)
        return correlation.save_reasoning_trace_correlation(mapping, path, **kwargs)

    # --- Animations ---

    def save_trajectory_animation(
        self,
        mapping: TrajectoryMapping | None = None,
        filename: str = "trajectory_maturation.gif",
        **kwargs,
    ) -> str:
        mapping = mapping or self._mapping
        if mapping is None:
            raise ValueError("Call trajectory_mapping() first or pass mapping=")
        path = str(self._ensure_dir() / filename)
        animate.animate_trajectory_maturation(mapping, save_path=path, **kwargs)
        return path

    def save_subspace_animation(
        self,
        prompt: str,
        *,
        trace: IntentTrace | None = None,
        filename: str = "reasoning_subspace.gif",
        **kwargs,
    ) -> str:
        trace = trace or self._trace
        path = str(self._ensure_dir() / filename)
        animate.animate_reasoning_subspace(
            self.bundle,
            prompt,
            trace=trace,
            save_path=path,
            **kwargs,
        )
        return path

    def save_intent_animation(
        self,
        trace: IntentTrace | None = None,
        filename: str = "intent_layers.gif",
        **kwargs,
    ) -> str:
        trace = trace or self._trace
        if trace is None:
            raise ValueError("Call intent_trace() first or pass trace=")
        path = str(self._ensure_dir() / filename)
        animate.animate_intent_grid(trace, save_path=path, **kwargs)
        return path

    def render_full_report(
        self,
        prompt: str,
        *,
        twin_b: str | None = None,
        concepts: list[str] | None = None,
        block_semantics: dict | None = None,
    ) -> dict[str, str]:
        """Generate all maps, correlations, and animations; return file paths."""
        mapping = self.trajectory_mapping(prompt, twin_b=twin_b, concepts=concepts)
        trace = self.intent_trace(prompt)

        paths = {
            "trajectory_map": self.save_trajectory_map(mapping),
            "reasoning_subspace": self.save_reasoning_subspace(prompt, mapping=mapping),
            "concept_correlation": self.save_concept_correlation(mapping),
            "reasoning_correlation": self.save_reasoning_correlation(mapping),
            "trajectory_animation": self.save_trajectory_animation(mapping),
            "subspace_animation": self.save_subspace_animation(prompt, trace=trace),
            "intent_animation": self.save_intent_animation(trace),
        }
        if block_semantics:
            paths["morpheme_map"] = self.save_morpheme_map(block_semantics)
        return paths
