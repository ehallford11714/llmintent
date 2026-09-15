"""Write a locked BoundState into the residual stream.

The beside-overlay failed: last-token paint did not change the mouth.
This writes the bound vector at the entity positions and every later
position, after every block, so attention and the MLP see 8~2 as part of
those tokens. Residual connections carry it. Decay is not trained.

This still does not claim the base weights grew object files.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from llmintent.models import ModelBundle, get_transformer_layers


def bind_positions(n_tokens: int, spans: list[dict[str, Any]]) -> list[int]:
    """Member birth sites plus every token after the last member."""
    if n_tokens < 1:
        return []
    if not spans:
        return [n_tokens - 1]
    sites = sorted({int(s["position"]) for s in spans if 0 <= int(s["position"]) < n_tokens})
    if not sites:
        return [n_tokens - 1]
    # From the first bound mention through the last token. Later layers
    # attend over a span that already carries the lock.
    return list(range(sites[0], n_tokens))


def _apply_to_hidden(
    hidden: torch.Tensor,
    vec: torch.Tensor,
    positions: list[int],
    gain: float,
) -> torch.Tensor:
    h = hidden.clone()
    v = vec.to(device=h.device, dtype=h.dtype).reshape(-1)
    if v.numel() != h.shape[-1]:
        raise ValueError(f"bound dim {v.numel()} != hidden {h.shape[-1]}")
    pos = [p for p in positions if 0 <= p < h.shape[1]]
    if not pos:
        return h
    h[:, pos, :] = h[:, pos, :] + float(gain) * v
    return h


@contextmanager
def bind_within_stream(
    bundle: ModelBundle,
    vector: torch.Tensor,
    positions: list[int],
    *,
    gain: float = 1.5,
) -> Iterator[None]:
    """After each block, add ``gain/n_layers * vector`` at ``positions``.

    Per-layer scale keeps LayerNorm from erasing a single write without
    summing a full ``gain`` sixty times.
    """
    layers = get_transformer_layers(bundle.model)
    n = max(1, len(layers))
    coeff = float(gain) / float(n)
    vec = vector.detach().float().reshape(-1)
    handles = []

    def _hook(_module, _inp, out):
        if isinstance(out, tuple):
            hidden = _apply_to_hidden(out[0], vec, positions, coeff)
            return (hidden,) + out[1:]
        return _apply_to_hidden(out, vec, positions, coeff)

    for layer in layers:
        handles.append(layer.register_forward_hook(_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def forward_within(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    vector: torch.Tensor,
    positions: list[int],
    *,
    gain: float = 1.5,
) -> list[torch.Tensor]:
    """One forward with the bound state written into the stream."""
    from llmintent.forward import forward_hidden_states_from_ids

    with bind_within_stream(bundle, vector, positions, gain=gain):
        return forward_hidden_states_from_ids(bundle, input_ids)
