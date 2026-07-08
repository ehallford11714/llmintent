"""Fit layer transport maps J_l approximating h_final ≈ J_l @ h_l (Jacobian lens proxy)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llmintent.forward import forward_hidden_states
from llmintent.models import ModelBundle


@dataclass
class TransportMaps:
    """Per-layer linear transport matrices toward the final residual basis."""

    maps: dict[int, torch.Tensor]
    num_layers: int
    num_samples: int

    def get(self, layer: int) -> torch.Tensor | None:
        return self.maps.get(layer)


def fit_transport_maps(
    bundle: ModelBundle,
    prompts: list[str],
    *,
    max_samples: int = 512,
    positions: str = "last",
) -> TransportMaps:
    """
    Estimate transport maps via least squares: h_final ≈ h_layer @ J_l.T

    Lightweight proxy for the Anthropic Jacobian E[∂h_final/∂h_l].
    """
    layer_samples: dict[int, list[torch.Tensor]] = {}
    final_samples: list[torch.Tensor] = []

    for text in prompts:
        _, states = forward_hidden_states(bundle, text)
        final = states[-1]
        if positions == "last":
            final_samples.append(final[0, -1, :].cpu())
            for layer_idx in range(len(states) - 1):
                layer_samples.setdefault(layer_idx, []).append(states[layer_idx][0, -1, :].cpu())
        else:
            seq_len = final.shape[1]
            for pos in range(seq_len):
                final_samples.append(final[0, pos, :].cpu())
                for layer_idx in range(len(states) - 1):
                    layer_samples.setdefault(layer_idx, []).append(states[layer_idx][0, pos, :].cpu())
                if len(final_samples) >= max_samples:
                    break
        if len(final_samples) >= max_samples:
            break

    maps: dict[int, torch.Tensor] = {}
    y = torch.stack(final_samples[:max_samples]).float()

    for layer_idx, samples in layer_samples.items():
        x = torch.stack(samples[: len(y)]).float()
        if x.shape[0] < x.shape[1]:
            # Underdetermined: use pseudo-inverse
            j_t = torch.linalg.lstsq(x, y).solution  # [hidden, hidden]
        else:
            j_t = torch.linalg.lstsq(x, y).solution
        maps[layer_idx] = j_t

    return TransportMaps(
        maps=maps,
        num_layers=bundle.num_layers,
        num_samples=len(y),
    )
