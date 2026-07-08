"""Retracement Transformer configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RetracementMode(str, Enum):
    """Retracement architecture variants for ablation."""

    BASELINE = "baseline"
    FOCUS_GATE = "focus_gate"
    RETRACE_STEER = "retrace_steer"
    DUAL_PASS = "dual_pass"
    WORKSPACE_LOOP = "workspace_loop"
    EXTREME = "extreme"


@dataclass
class RetracementConfig:
    """
    Architecture hyperparameters for the Retracement Transformer.

    Depth partition (from J-space / activation insights):
      sensory  → layers [0, sensory_end)
      workspace → layers [sensory_end, workspace_end)  ← retrace injection
      motor    → layers [workspace_end, num_layers)
    """

    mode: RetracementMode = RetracementMode.FOCUS_GATE
    sensory_fraction: float = 0.33
    workspace_fraction: float = 0.45
    focus_coefficient: float = 0.35
    retrace_coefficient: float = 0.5
    dual_pass_blend: float = 0.4
    extreme_passes: int = 2
    gate_temperature: float = 1.0
    use_dynamic_pivot: bool = True
    pivot_layer: int | None = None
    workspace_layers: list[int] = field(default_factory=list)

    def partition(self, num_layers: int) -> tuple[int, int, int]:
        """Return (sensory_end, workspace_end, num_layers) layer indices."""
        sensory_end = max(1, int(num_layers * self.sensory_fraction))
        workspace_end = max(
            sensory_end + 1,
            int(num_layers * (self.sensory_fraction + self.workspace_fraction)),
        )
        workspace_end = min(workspace_end, num_layers - 1)
        return sensory_end, workspace_end, num_layers

    def resolve_layers(self, num_layers: int) -> list[int]:
        """Layers where retrace blocks attach."""
        if self.workspace_layers:
            return self.workspace_layers
        sensory_end, workspace_end, _ = self.partition(num_layers)
        if self.mode == RetracementMode.BASELINE:
            return []
        if self.mode in (RetracementMode.FOCUS_GATE, RetracementMode.RETRACE_STEER):
            pivot = self.pivot_layer if self.pivot_layer is not None else sensory_end
            return [min(max(0, pivot), num_layers - 1)]
        if self.mode == RetracementMode.DUAL_PASS:
            return list(range(sensory_end, workspace_end))
        if self.mode in (RetracementMode.WORKSPACE_LOOP, RetracementMode.EXTREME):
            return list(range(sensory_end, workspace_end))
        return [sensory_end]
