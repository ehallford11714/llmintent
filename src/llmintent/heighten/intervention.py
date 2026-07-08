"""Activation-level focus steering at reasoning layers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from llmintent.cognitive.types import CognitiveModuleProfile
from llmintent.forward import forward_hidden_states, get_lm_head, normalize_hidden
from llmintent.heighten.focus import compare_focus, compute_focus_metrics
from llmintent.heighten.types import FocusMetrics
from llmintent.kernels.kl_kernel import collect_twin_hidden_matrix
from llmintent.models import ModelBundle, get_transformer_layers
from llmintent.trajectory import build_trajectory_mapping


@dataclass
class SteeringResult:
    baseline_focus: FocusMetrics
    steered_focus: FocusMetrics
    focus_gain: dict[str, float]
    layers_steered: list[int]
    coefficient: float

    def to_dict(self) -> dict:
        return {
            "baseline_focus": self.baseline_focus.to_dict(),
            "steered_focus": self.steered_focus.to_dict(),
            "focus_gain": compare_focus(self.baseline_focus, self.steered_focus),
            "layers_steered": self.layers_steered,
            "coefficient": self.coefficient,
        }


def extract_reasoning_focus_vector(
    bundle: ModelBundle,
    anchor_prompt: str,
    retrace_prompt: str,
    *,
    cognitive: CognitiveModuleProfile | None = None,
    position: int = -1,
) -> torch.Tensor:
    """
    Direction in hidden space from diffuse → focused reasoning.

    Uses twin hidden delta (retrace − anchor), optionally scaled by reasoning kernel.
    """
    h_a, h_b = collect_twin_hidden_matrix(bundle, anchor_prompt, retrace_prompt, position=position)
    delta = (h_b - h_a).mean(dim=0)
    vec = delta / (torch.norm(delta) + 1e-8)

    if cognitive is not None:
        reasoning_kernel = next((k for k in cognitive.kernels if k.module == "reasoning"), None)
        if reasoning_kernel is not None and reasoning_kernel.kernel_basis.numel() > 0:
            basis = reasoning_kernel.kernel_basis[0].float()
            basis = basis / (torch.norm(basis) + 1e-8)
            vec = 0.6 * vec + 0.4 * basis
            vec = vec / (torch.norm(vec) + 1e-8)

    return vec


@contextmanager
def steering_hooks(
    bundle: ModelBundle,
    layer_indices: list[int],
    steering_vector: torch.Tensor,
    coefficient: float,
):
    """Register forward hooks that add steering_vector to the last token hidden state."""
    layers = get_transformer_layers(bundle.model)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    vec = steering_vector.float()

    def _make_hook():
        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                hidden = out[0]
                hidden = hidden.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + coefficient * vec.to(hidden.device)
                return (hidden,) + out[1:]
            hidden = out.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + coefficient * vec.to(hidden.device)
            return hidden

        return hook

    for idx in layer_indices:
        idx = max(0, min(idx, len(layers) - 1))
        handles.append(layers[idx].register_forward_hook(_make_hook()))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# Backward-compatible alias
_steering_hooks = steering_hooks


def forward_with_focus_steering(
    bundle: ModelBundle,
    text: str,
    layer_indices: list[int],
    steering_vector: torch.Tensor,
    *,
    coefficient: float = 0.5,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Forward pass with reasoning focus vector injected at selected layers.

    Returns (hidden_states, final_logits_at_last_token).
    """
    with steering_hooks(bundle, layer_indices, steering_vector, coefficient):
        _, states = forward_hidden_states(bundle, text)
    head = get_lm_head(bundle)
    last = normalize_hidden(bundle, states[-1][0, -1, :].float())
    logits = head(last.to(bundle.device))
    return states, logits


def apply_focus_steering(
    bundle: ModelBundle,
    prompt: str,
    anchor_prompt: str,
    retrace_prompt: str,
    *,
    layer_indices: list[int],
    concepts: list[str] | None = None,
    coefficient: float = 0.5,
    transport=None,
    cognitive: CognitiveModuleProfile | None = None,
) -> SteeringResult:
    """
    Measure focus gain from activation steering toward retrace direction.

    Compares trajectory focus before/after steering at reasoning layers.
    """
    baseline_mapping = build_trajectory_mapping(
        bundle,
        prompt,
        twin_b=anchor_prompt,
        transport=transport,
        concepts=concepts,
    )
    baseline_focus = compute_focus_metrics(baseline_mapping)

    vec = extract_reasoning_focus_vector(
        bundle,
        anchor_prompt,
        retrace_prompt,
        cognitive=cognitive,
        position=-1,
    )

    with steering_hooks(bundle, layer_indices, vec, coefficient):
        steered_mapping = build_trajectory_mapping(
            bundle,
            prompt,
            twin_b=retrace_prompt,
            transport=transport,
            concepts=concepts,
        )
    steered_focus = compute_focus_metrics(steered_mapping)

    return SteeringResult(
        baseline_focus=baseline_focus,
        steered_focus=steered_focus,
        focus_gain=compare_focus(baseline_focus, steered_focus),
        layers_steered=layer_indices,
        coefficient=coefficient,
    )
