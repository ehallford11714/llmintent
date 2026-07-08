"""KL + twin Barlow kernel exports."""

from llmintent.kernels.barlow import (
    barlow_cross_correlation,
    barlow_invariance_score,
    barlow_redundancy_score,
    barlow_twins_loss,
    extract_kernel_basis,
    minimize_twin_barlow,
    per_layer_barlow_features,
)
from llmintent.kernels.kl_kernel import (
    collect_twin_hidden_matrix,
    kl_weighted_difference_kernel,
    per_layer_kl_profile,
)

__all__ = [
    "barlow_cross_correlation",
    "barlow_invariance_score",
    "barlow_redundancy_score",
    "barlow_twins_loss",
    "collect_twin_hidden_matrix",
    "extract_kernel_basis",
    "kl_weighted_difference_kernel",
    "minimize_twin_barlow",
    "per_layer_barlow_features",
    "per_layer_kl_profile",
]
