"""Retracement blocks: focus gate and retrace merge."""

from __future__ import annotations

import torch
import torch.nn as nn


class FocusGate(nn.Module):
    """
    Gated injection of a focus direction into hidden states.

    h' = h + coeff * sigmoid(cos(h, v) / temperature) * v

    Implements concentration of computation along a reasoning axis without
    retraining base transformer weights.
    """

    def __init__(self, hidden_size: int, coefficient: float = 0.35, temperature: float = 1.0):
        super().__init__()
        self.coefficient = coefficient
        self.temperature = temperature
        self.register_buffer("focus_vector", torch.zeros(hidden_size))

    def set_focus_vector(self, vec: torch.Tensor) -> None:
        vec = vec.float().detach()
        vec = vec / (torch.norm(vec) + 1e-8)
        self.focus_vector.copy_(vec.to(self.focus_vector.device))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        v = self.focus_vector.to(hidden.dtype)
        h = hidden
        # Cosine-like gate per token position
        h_norm = torch.norm(h, dim=-1, keepdim=True).clamp(min=1e-8)
        v_exp = v.view(1, 1, -1)
        cos = (h * v_exp).sum(dim=-1, keepdim=True) / h_norm
        gate = torch.sigmoid(cos / self.temperature)
        return h + self.coefficient * gate * v_exp


class RetraceMerge(nn.Module):
    """
    Blend current hidden state with a retrace snapshot.

    h' = (1 - blend) * h + blend * h_retrace

    Used in dual-pass / workspace-loop modes.
    """

    def __init__(self, blend: float = 0.4):
        super().__init__()
        self.blend = blend
        self.register_buffer("retrace_snapshot", torch.zeros(1))

    def set_snapshot(self, hidden: torch.Tensor) -> None:
        self.retrace_snapshot = hidden.detach().clone()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        snap = self.retrace_snapshot.to(hidden.dtype)
        if snap.shape != hidden.shape:
            return hidden
        return (1.0 - self.blend) * hidden + self.blend * snap


def compute_self_focus_vector(hidden: torch.Tensor) -> torch.Tensor:
    """
    Derive focus direction from hidden states at pivot (self-retrace signal).

    Uses mean hidden state contrasted with layer dispersion direction.
    """
    # hidden: [batch, seq, dim]
    mean_h = hidden.mean(dim=1)  # [batch, dim]
    centered = hidden - mean_h.unsqueeze(1)
    dispersion = centered.mean(dim=1)
    focus = mean_h - 0.5 * dispersion
    focus = focus.mean(dim=0)
    return focus / (torch.norm(focus) + 1e-8)
