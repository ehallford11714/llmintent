"""Retracement Transformer forward pass and hook wiring."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from llmintent.forward import get_lm_head
from llmintent.heighten.intervention import extract_reasoning_focus_vector
from llmintent.models import ModelBundle, get_transformer_layers
from llmintent.retracement.blocks import FocusGate, RetraceMerge, compute_self_focus_vector
from llmintent.retracement.config import RetracementConfig, RetracementMode


@dataclass
class RetracementState:
    """Runtime state captured during a retracement forward pass."""

    pivot_layer: int
    workspace_layers: list[int]
    focus_vector: torch.Tensor | None = None
    mode: str = ""
    hook_count: int = 0
    metadata: dict = field(default_factory=dict)


class RetracementTransformer:
    """
    Retracement Transformer — focused-reasoning architecture wrapper.

    Proposed structure (inference-time, frozen backbone):

    ```text
    Input → [Sensory layers] → RETRACE PIVOT (FocusGate)
          → [Workspace layers] → optional dual-pass merge
          → [Motor layers] → LM head
    ```

    Modes:
    - baseline: standard forward (ablation control)
    - focus_gate: sigmoid-gated focus vector at pivot
    - retrace_steer: external anchor→retrace delta injection
    - dual_pass: snapshot at pivot, merge in workspace band
    - workspace_loop: focus gate at every workspace layer
    - extreme: workspace_loop + amplified coefficients + multi-pass blend
    """

    def __init__(
        self,
        bundle: ModelBundle,
        config: RetracementConfig | None = None,
        *,
        anchor_text: str | None = None,
        retrace_text: str | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = config or RetracementConfig()
        self.anchor_text = anchor_text
        self.retrace_text = retrace_text
        hidden = bundle.hidden_size
        self.focus_gate = FocusGate(
            hidden,
            coefficient=self.config.focus_coefficient,
            temperature=self.config.gate_temperature,
        ).to(bundle.device)
        self.retrace_merge = RetraceMerge(blend=self.config.dual_pass_blend).to(bundle.device)
        self._state = RetracementState(pivot_layer=0, workspace_layers=[])

    def _resolve_focus_vector(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.anchor_text and self.retrace_text:
            vec = extract_reasoning_focus_vector(
                self.bundle,
                self.anchor_text,
                self.retrace_text,
            )
            return vec.to(self.bundle.device)

        # Self-retrace: probe pivot hidden from partial forward
        with torch.no_grad():
            outputs = self.bundle.model(input_ids, output_hidden_states=True)
            states = outputs.hidden_states
            sensory_end, _, _ = self.config.partition(len(states) - 1)
            pivot = self.config.pivot_layer or sensory_end
            pivot = min(pivot, len(states) - 1)
            h = states[pivot]
            return compute_self_focus_vector(h).to(self.bundle.device)

    def _build_hooks(self, focus_vec: torch.Tensor, snapshot: torch.Tensor | None):
        layers = get_transformer_layers(self.bundle.model)
        cfg = self.config
        layer_indices = cfg.resolve_layers(self.bundle.num_layers)
        handles = []
        self.focus_gate.set_focus_vector(focus_vec)
        if snapshot is not None:
            self.retrace_merge.set_snapshot(snapshot)

        coeff = cfg.retrace_coefficient
        if cfg.mode == RetracementMode.EXTREME:
            coeff *= cfg.extreme_passes

        def _make_focus_hook():
            def hook(_mod, _inp, out):
                if isinstance(out, tuple):
                    h = out[0].clone()
                    h = self.focus_gate(h)
                    if snapshot is not None and cfg.mode in (
                        RetracementMode.DUAL_PASS,
                        RetracementMode.WORKSPACE_LOOP,
                        RetracementMode.EXTREME,
                    ):
                        h = self.retrace_merge(h)
                    return (h,) + out[1:]
                h = out.clone()
                h = self.focus_gate(h)
                return h

            return hook

        def _make_steer_hook():
            vec = focus_vec.float()

            def hook(_mod, _inp, out):
                if isinstance(out, tuple):
                    h = out[0].clone()
                    h[:, -1, :] = h[:, -1, :] + coeff * vec.to(h.device)
                    return (h,) + out[1:]
                h = out.clone()
                h[:, -1, :] = h[:, -1, :] + coeff * vec.to(h.device)
                return h

            return hook

        hook_fn = _make_steer_hook if cfg.mode == RetracementMode.RETRACE_STEER else _make_focus_hook

        for idx in layer_indices:
            idx = max(0, min(idx, len(layers) - 1))
            handles.append(layers[idx].register_forward_hook(hook_fn()))

        self._state.hook_count = len(handles)
        self._state.workspace_layers = layer_indices
        return handles

    @contextmanager
    def _forward_context(self, input_ids: torch.Tensor):
        if self.config.mode == RetracementMode.BASELINE:
            yield
            return

        focus_vec = self._resolve_focus_vector(input_ids)
        self._state.focus_vector = focus_vec
        self._state.mode = self.config.mode.value

        snapshot = None
        if self.config.mode in (
            RetracementMode.DUAL_PASS,
            RetracementMode.WORKSPACE_LOOP,
            RetracementMode.EXTREME,
        ):
            with torch.no_grad():
                out = self.bundle.model(input_ids, output_hidden_states=True)
                sensory_end, _, _ = self.config.partition(len(out.hidden_states) - 1)
                pivot = min(sensory_end, len(out.hidden_states) - 1)
                snapshot = out.hidden_states[pivot].clone()
                self._state.pivot_layer = pivot

        handles = self._build_hooks(focus_vec, snapshot)
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning logits [batch, seq, vocab]."""
        with self._forward_context(input_ids):
            outputs = self.bundle.model(input_ids)
            return outputs.logits if hasattr(outputs, "logits") else outputs[0]

    def token_loss(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Causal LM cross-entropy; returns (total_nll, num_tokens)."""
        labels = input_ids.clone()
        with self._forward_context(input_ids):
            outputs = self.bundle.model(input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None

        if loss is not None:
            n_tokens = (labels[:, 1:] != -100).sum().item() if labels.numel() else labels.numel()
            return loss * max(n_tokens, 1), max(n_tokens, 1)

        logits = self.forward(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        return nll, shift_labels.numel()

    @property
    def state(self) -> RetracementState:
        return self._state
