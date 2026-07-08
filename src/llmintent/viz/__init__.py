"""Visualization tooling: maps, correlation matrices, animations."""

from llmintent.viz.animate import (
    animate_intent_grid,
    animate_reasoning_subspace,
    animate_trajectory_maturation,
)
from llmintent.viz.correlation import (
    build_concept_correlation_matrix,
    build_reasoning_trace_correlation,
    plot_concept_correlation,
    plot_correlation_matrix,
    plot_reasoning_trace_correlation,
)
from llmintent.viz.maps import (
    plot_morpheme_map,
    plot_reasoning_subspace,
    plot_trajectory_map,
)
from llmintent.viz.suite import VisualizationSuite

__all__ = [
    "VisualizationSuite",
    "animate_intent_grid",
    "animate_reasoning_subspace",
    "animate_trajectory_maturation",
    "build_concept_correlation_matrix",
    "build_reasoning_trace_correlation",
    "plot_concept_correlation",
    "plot_correlation_matrix",
    "plot_morpheme_map",
    "plot_reasoning_subspace",
    "plot_reasoning_trace_correlation",
    "plot_trajectory_map",
]
