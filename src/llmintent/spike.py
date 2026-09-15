"""Parallel spike bank: transience beside a transformer, not a trained memory.

A transformer residual does not persist an entity from token t to token t+1
(Tang, Zhao et al., ICML 2026: sequential tasks solved non-sequentially at the
query token). This module is a leaky integrate-and-fire bank laid *beside*
that stack. When a residual node is active, it injects current here in the
same clock tick. Voltage and eligibility traces then decay at a fixed rate.

Decay is a Python float, never an ``nn.Parameter``. Training it would let the
net learn to hold or dump state. The leak is the point: bindings are live
only while traces remain above floor, then they expire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpikeConfig:
    """Fixed physiology. None of these are trained."""

    decay: float = 0.92
    threshold: float = 1.0
    reset: float = 0.0
    fire_k: float = 1.0
    max_nodes: int = 256
    max_named: int = 32
    entity_amp: float = 2.0
    bind_floor: float = 0.05
    live_floor: float = 0.15
    max_bindings: int = 64

    def __post_init__(self) -> None:
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {self.decay}")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")


class SpikePotentialBank(torch.nn.Module):
    """Vector LIF. State is voltage + trace; decay cannot be optimized."""

    def __init__(self, n_cells: int, config: SpikeConfig | None = None) -> None:
        super().__init__()
        if n_cells < 1:
            raise ValueError("n_cells must be >= 1")
        cfg = config or SpikeConfig()
        self.config = cfg
        self.n_cells = int(n_cells)
        # Frozen hyperparameter — not a buffer, not a Parameter.
        self._decay = float(cfg.decay)
        self._threshold = float(cfg.threshold)
        self._reset = float(cfg.reset)
        self.register_buffer("v", torch.zeros(self.n_cells), persistent=True)
        self.register_buffer("trace", torch.zeros(self.n_cells), persistent=True)
        self.register_buffer("spike_count", torch.zeros(self.n_cells), persistent=True)
        self.clock = 0
        self.eval()

    @property
    def decay(self) -> float:
        return self._decay

    def train(self, mode: bool = True) -> SpikePotentialBank:  # type: ignore[override]
        """Stay in eval. A train() flag must not create a learning path."""
        return super().train(False)

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        # Empty: Adam(bank.parameters()) cannot see decay.
        return iter(())

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return iter(())

    def reset_state(self) -> None:
        self.v.zero_()
        self.trace.zero_()
        self.spike_count.zero_()
        self.clock = 0

    def step(self, current: torch.Tensor) -> torch.Tensor:
        """One token-clock tick. ``current`` injects; then leak; then fire."""
        x = current.detach().float().reshape(-1)
        if x.numel() != self.n_cells:
            raise ValueError(f"current has {x.numel()} cells, bank has {self.n_cells}")
        if x.device != self.v.device:
            self.v = self.v.to(x.device)
            self.trace = self.trace.to(x.device)
            self.spike_count = self.spike_count.to(x.device)
        # Integrate, then leak. Decay multiplies the *held* state.
        v = self._decay * self.v + x
        spikes = (v >= self._threshold).to(v.dtype)
        v = torch.where(spikes > 0, torch.full_like(v, self._reset), v)
        self.v = v
        self.trace = self._decay * self.trace + spikes
        self.spike_count = self.spike_count + spikes
        self.clock += 1
        return spikes

    def idle(self, n: int = 1) -> torch.Tensor:
        """Time passes with no transformer activation. Pure transience."""
        last = torch.zeros(self.n_cells, device=self.v.device)
        for _ in range(max(0, int(n))):
            last = self.step(torch.zeros(self.n_cells, device=self.v.device))
        return last

    def live_mask(self, floor: float | None = None) -> torch.Tensor:
        fl = self.config.live_floor if floor is None else float(floor)
        return self.trace >= fl

    def snapshot(self) -> dict[str, Any]:
        return {
            "clock": self.clock,
            "decay": self._decay,
            "n_cells": self.n_cells,
            "trainable_params": 0,
            "v_max": float(self.v.max().item()) if self.v.numel() else 0.0,
            "trace_max": float(self.trace.max().item()) if self.trace.numel() else 0.0,
            "n_live": int(self.live_mask().sum().item()),
            "n_ever_spiked": int((self.spike_count > 0).sum().item()),
        }


def node_current(
    hidden: torch.Tensor,
    *,
    channel_index: torch.Tensor,
    fire_k: float,
) -> torch.Tensor:
    """Fixed map: residual channels → LIF current. No learned projection.

    A node is 'activated' when its channel exceeds ``fire_k`` residual RMS.
    """
    h = hidden.detach().float().reshape(-1)
    idx = channel_index.to(device=h.device)
    x = h[idx]
    rms = x.pow(2).mean().sqrt().clamp_min(1e-8)
    return F.relu(x / rms - float(fire_k))


def node_adjoint(
    hidden: torch.Tensor,
    *,
    channel_index: torch.Tensor,
    node_traces: torch.Tensor,
    fire_k: float,
) -> torch.Tensor:
    """Surrogate ∇_h (trace · current) with traces held fixed.

    Hard spikes have zero gradient almost everywhere. This is the ReLU-current
    adjoint, not a trained backward. Chronic channels with live traces still
    dominate if you scatter the whole node bank — that is the 27B soup.
    """
    h = hidden.detach().float().reshape(-1)
    idx = channel_index.to(device=h.device)
    x = h[idx]
    rms = x.pow(2).mean().sqrt().clamp_min(1e-8)
    gate = (x / rms - float(fire_k)) > 0
    tr = node_traces.detach().float().reshape(-1).to(device=h.device)
    if tr.numel() != idx.numel():
        raise ValueError(f"node_traces has {tr.numel()} cells, index has {idx.numel()}")
    contrib = tr * gate.to(dtype=h.dtype) / rms
    g = torch.zeros_like(h)
    g[idx] = contrib
    return g
