"""Persistence binding: lock the state; traces only mark liveness.

The transformer stays the next-token engine. A parallel LIF bank spikes when
named entities are born. The named adjoint g is written into a BoundState and
locked. Last-token residuals do not overwrite it. Spike-trace decay is
transience of *access*, not of the bound content. Decay is not trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from llmintent.spike import SpikeConfig, SpikePotentialBank, node_current


@dataclass
class Binding:
    """Undirected glue between two cells, held only while traces last."""

    left: int
    right: int
    left_name: str
    right_name: str
    potential: float
    born_at: int
    last_spike: int

    def key(self) -> tuple[int, int]:
        a, b = sorted((self.left, self.right))
        return a, b

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left_name,
            "right": self.right_name,
            "potential": round(float(self.potential), 4),
            "born_at": self.born_at,
            "last_spike": self.last_spike,
            "cells": [self.left, self.right],
        }


@dataclass
class BoundState:
    """Locked state register. Vector is written once the member set is bound."""

    id: str
    members: tuple[str, ...]
    vector: torch.Tensor
    scale: float
    born_at: int
    locked: bool = False
    live: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "members": list(self.members),
            "dim": int(self.vector.numel()),
            "scale": round(float(self.scale), 4),
            "born_at": self.born_at,
            "locked": self.locked,
            "live": self.live,
        }


@dataclass
class BindReport:
    text: str
    last_token: str
    decay: float
    trainable_params: int
    named_traces: dict[str, float]
    live_bindings: list[Binding]
    expired: list[str]
    history: list[dict[str, Any]]
    mouth_baseline: str
    mouth_with_bindings: str
    bound_scale: float = 0.0
    bound_state: BoundState | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "last_token": self.last_token,
            "decay": self.decay,
            "trainable_params": self.trainable_params,
            "named_traces": {k: round(v, 4) for k, v in self.named_traces.items()},
            "live_bindings": [b.to_dict() for b in self.live_bindings],
            "expired": list(self.expired),
            "bound_scale": round(float(self.bound_scale), 4),
            "bound_state": None if self.bound_state is None else self.bound_state.to_dict(),
            "history": self.history,
            "mouth_baseline": self.mouth_baseline,
            "mouth_with_bindings": self.mouth_with_bindings,
            "notes": list(self.notes),
        }


class PersistenceBinder:
    """Token-clock binder. Transformer hidden states drive a frozen LIF bank."""

    def __init__(self, config: SpikeConfig | None = None) -> None:
        self.config = config or SpikeConfig()
        self.bank: SpikePotentialBank | None = None
        self.n_nodes = 0
        self.channel_index: torch.Tensor | None = None
        self.name_to_cell: dict[str, int] = {}
        self.cell_to_name: dict[int, str] = {}
        self.payload: dict[int, torch.Tensor] = {}
        self.bindings: dict[tuple[int, int], Binding] = {}
        self.expired: list[str] = []
        self.history: list[dict[str, Any]] = []
        self._layer_current: torch.Tensor | None = None
        self._layers_seen = 0
        self.mentions = 0
        self.bound_state: BoundState | None = None

    def reset(self) -> None:
        if self.bank is not None:
            self.bank.reset_state()
        self.name_to_cell.clear()
        self.cell_to_name.clear()
        self.payload.clear()
        self.bindings.clear()
        self.expired.clear()
        self.history.clear()
        self._layer_current = None
        self._layers_seen = 0
        self.mentions = 0
        self.bound_state = None

    @property
    def bound_grad(self) -> torch.Tensor | None:
        return None if self.bound_state is None else self.bound_state.vector

    @property
    def bound_scale(self) -> float:
        return 0.0 if self.bound_state is None else float(self.bound_state.scale)

    def _ensure(self, dim: int, device: torch.device) -> SpikePotentialBank:
        if self.bank is not None:
            return self.bank
        n_nodes = min(int(self.config.max_nodes), int(dim))
        n_cells = n_nodes + int(self.config.max_named)
        self.n_nodes = n_nodes
        self.channel_index = torch.linspace(0, dim - 1, steps=n_nodes).long()
        self.bank = SpikePotentialBank(n_cells, self.config)
        self.bank.to(device)
        return self.bank

    def _named_cell(self, name: str) -> int | None:
        if name in self.name_to_cell:
            return self.name_to_cell[name]
        used = len(self.name_to_cell)
        if used >= int(self.config.max_named):
            return None
        idx = self.n_nodes + used
        self.name_to_cell[name] = idx
        self.cell_to_name[idx] = name
        return idx

    def begin_token(self) -> None:
        self._layer_current = None
        self._layers_seen = 0

    def accumulate_layer(self, hidden: torch.Tensor) -> None:
        """Call once per transformer layer at the current token (parallel inject)."""
        h = hidden.detach().float().reshape(-1)
        bank = self._ensure(h.numel(), h.device)
        assert self.channel_index is not None
        cur = node_current(h, channel_index=self.channel_index, fire_k=self.config.fire_k)
        if self._layer_current is None:
            self._layer_current = cur
        else:
            self._layer_current = self._layer_current + cur
        self._layers_seen += 1

    def end_token(
        self,
        *,
        token: str,
        residual: torch.Tensor,
        entity_name: str | None = None,
    ) -> torch.Tensor:
        """One leak after all layers of this token have injected."""
        if self.bank is None or self._layer_current is None:
            h = residual.detach().float().reshape(-1)
            self._ensure(h.numel(), h.device)
            self.accumulate_layer(h)
        assert self.bank is not None and self._layer_current is not None
        n_layers = max(1, self._layers_seen)
        current = torch.zeros(self.bank.n_cells, device=self.bank.v.device)
        current[: self.n_nodes] = self._layer_current / float(n_layers)
        if entity_name:
            self.mentions += 1
            idx = self._named_cell(entity_name)
            if idx is not None:
                current[idx] = float(self.config.entity_amp)
        spikes = self.bank.step(current)
        self._record_payloads(spikes, residual)
        self._update_bindings(spikes)
        self._bind_state()
        self._snapshot_history(token)
        self._layer_current = None
        self._layers_seen = 0
        return spikes

    def idle(self, n: int = 1) -> None:
        if self.bank is None:
            return
        for _ in range(max(0, int(n))):
            self.bank.idle(1)
            self._update_bindings(torch.zeros(self.bank.n_cells, device=self.bank.v.device))
            self._bind_state()
            self._snapshot_history("")

    def _record_payloads(self, spikes: torch.Tensor, residual: torch.Tensor) -> None:
        """Lock named-cell residuals at first spike. Do not follow the last token."""
        h = residual.detach().float().reshape(-1).cpu()
        for idx in self.name_to_cell.values():
            if idx < spikes.numel() and float(spikes[idx]) > 0 and idx not in self.payload:
                self.payload[int(idx)] = h

    def _cell_name(self, idx: int) -> str:
        if idx in self.cell_to_name:
            return self.cell_to_name[idx]
        return f"node:{idx}"

    def _update_bindings(self, spikes: torch.Tensor) -> None:
        assert self.bank is not None
        decay = self.bank.decay
        clock = self.bank.clock
        for key, bind in list(self.bindings.items()):
            bind.potential *= decay
            if bind.potential < self.config.bind_floor:
                self.expired.append(f"{bind.left_name}~{bind.right_name}")
                del self.bindings[key]
        live = torch.nonzero(self.bank.live_mask(), as_tuple=False).flatten().tolist()
        spiked = torch.nonzero(spikes > 0, as_tuple=False).flatten().tolist()
        traces = self.bank.trace.detach().cpu()
        # Named cells bind first (readable object files); then spike-vs-live nodes.
        named = list(self.name_to_cell.values())
        candidates: list[tuple[int, int]] = []
        for a in named:
            for b in named:
                if a < b:
                    candidates.append((a, b))
        for i in spiked:
            for j in live:
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                candidates.append((a, b))
        seen: set[tuple[int, int]] = set()
        for a, b in candidates:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pot = float(traces[a].item() * traces[b].item())
            if pot < self.config.bind_floor:
                continue
            key = (a, b)
            if key in self.bindings:
                bind = self.bindings[key]
                bind.potential = max(bind.potential, pot)
                bind.last_spike = clock
            else:
                self.bindings[key] = Binding(
                    left=a,
                    right=b,
                    left_name=self._cell_name(a),
                    right_name=self._cell_name(b),
                    potential=pot,
                    born_at=clock,
                    last_spike=clock,
                )
        if len(self.bindings) > self.config.max_bindings:
            ranked = sorted(self.bindings.values(), key=lambda b: b.potential, reverse=True)
            keep = ranked[: self.config.max_bindings]
            self.bindings = {b.key(): b for b in keep}

    def _snapshot_history(self, token: str) -> None:
        assert self.bank is not None
        named = {n: float(self.bank.trace[i].item()) for n, i in self.name_to_cell.items()}
        live = [b for b in self.bindings.values() if b.potential >= self.config.live_floor]
        self.history.append(
            {
                "t": self.bank.clock,
                "token": token,
                "named_traces": {k: round(v, 4) for k, v in named.items()},
                "n_live_bindings": len(live),
                "top_binding": live[0].to_dict() if live else None,
                "bound_scale": round(float(self.bound_scale), 4),
                "bound_locked": bool(self.bound_state.locked) if self.bound_state else False,
                "bound_live": bool(self.bound_state.live) if self.bound_state else False,
            }
        )

    def live_bindings(self) -> list[Binding]:
        live = [b for b in self.bindings.values() if b.potential >= self.config.live_floor]
        live.sort(key=lambda b: b.potential, reverse=True)
        return live

    def named_traces(self) -> dict[str, float]:
        if self.bank is None:
            return {}
        return {n: float(self.bank.trace[i].item()) for n, i in self.name_to_cell.items()}

    def named_gradient(self) -> torch.Tensor | None:
        """∇_h of trace-weighted named alignment. Traces leak; payloads stay at birth.

        g = Σ_i trace_i · unit(payload_i). This is the spike-net gradient in
        residual space. Chronic node soup is not included.
        """
        if self.bank is None:
            return None
        acc = None
        traces = self.bank.trace.detach().float().cpu()
        for idx in self.name_to_cell.values():
            tr = float(traces[idx].item())
            vec = self.payload.get(idx)
            if vec is None or tr < self.config.bind_floor:
                continue
            u = vec / (torch.linalg.norm(vec) + 1e-8)
            acc = tr * u if acc is None else acc + tr * u
        return acc

    def _members_live(self, members: tuple[str, ...]) -> bool:
        if self.bank is None or not members:
            return False
        traces = self.named_traces()
        return all(traces.get(name, 0.0) >= self.config.live_floor for name in members)

    def _should_lock(self, members: tuple[str, ...]) -> bool:
        return len(members) >= 2 or self.mentions >= 2

    def _bind_state(self) -> None:
        """Write g; grow when a new name is born. Last token cannot overwrite."""
        if self.bank is None:
            return
        members = tuple(sorted(self.name_to_cell.keys()))
        live = self._members_live(members)
        grew = self.bound_state is not None and members != self.bound_state.members
        if self.bound_state is not None and self.bound_state.locked and not grew:
            self.bound_state.live = live
            return
        g = self.named_gradient()
        if g is None:
            if self.bound_state is not None:
                self.bound_state.live = live
            return
        n = float(torch.linalg.norm(g))
        if n <= 1e-8:
            return
        u = (g / n).detach().clone()
        clock = self.bank.clock
        if self.bound_state is None:
            self.bound_state = BoundState(
                id="~".join(members),
                members=members,
                vector=u,
                scale=n,
                born_at=clock,
                locked=self._should_lock(members),
                live=live,
            )
            return
        self.bound_state.id = "~".join(members)
        self.bound_state.members = members
        self.bound_state.vector = u
        self.bound_state.scale = n
        self.bound_state.live = live
        if self._should_lock(members):
            self.bound_state.locked = True

    def state_vector(self) -> torch.Tensor | None:
        """Locked bound state. Available even after traces fade."""
        if self.bound_state is None:
            return None
        v = self.bound_state.vector.reshape(-1)
        n = torch.linalg.norm(v)
        if float(n) < 1e-8:
            return None
        return v / n

    def bind_in(self, hidden: torch.Tensor, *, gain: float = 1.5) -> torch.Tensor:
        """Add the locked bound state to a residual. No-op if unbound."""
        vec = self.state_vector()
        h = hidden.detach().float().reshape(-1).cpu()
        if vec is None:
            return h
        v = vec.reshape(-1).to(dtype=h.dtype)
        if v.numel() != h.numel():
            raise ValueError(f"bound state dim {v.numel()} != residual dim {h.numel()}")
        return h + float(gain) * v

    def soup_vector(self) -> torch.Tensor | None:
        """Old join: live-binding residual mix. Dominated by chronic nodes on 27B."""
        live = self.live_bindings()
        if not live:
            return None
        acc = None
        wsum = 0.0
        for bind in live:
            for idx in (bind.left, bind.right):
                vec = self.payload.get(idx)
                if vec is None:
                    continue
                w = float(bind.potential)
                acc = vec * w if acc is None else acc + vec * w
                wsum += w
        if acc is None or wsum <= 0:
            return None
        v = acc / wsum
        n = torch.linalg.norm(v)
        if float(n) < 1e-8:
            return None
        return v / n

    def binding_vector(self) -> torch.Tensor | None:
        """Default inject: bound gradient. Falls back to soup only if unbound."""
        return self.state_vector() or self.soup_vector()


def drive_from_hidden(
    binder: PersistenceBinder,
    hidden_states: list[torch.Tensor] | torch.Tensor,
    *,
    tokens: list[str] | None = None,
    spans: list[dict[str, Any]] | None = None,
) -> PersistenceBinder:
    """Tick the binder once per token. Layers at that token inject in parallel."""
    if isinstance(hidden_states, torch.Tensor):
        layers = [hidden_states]
    else:
        layers = list(hidden_states)
    if not layers:
        raise ValueError("hidden_states is empty")
    first = layers[0]
    if first.dim() == 3:
        seq = first.shape[1]
        layers = [H[0] if H.dim() == 3 else H for H in layers]
    elif first.dim() == 2:
        seq = first.shape[0]
    else:
        raise ValueError(f"expected [T,D] or [B,T,D], got {tuple(first.shape)}")
    pos_name: dict[int, str] = {}
    for span in spans or []:
        pos_name[int(span["position"])] = f"{span['kind']}:{span['token']}"
    names = list(tokens) if tokens is not None else [""] * seq
    if len(names) < seq:
        names = names + [""] * (seq - len(names))
    for t in range(seq):
        binder.begin_token()
        last_h = None
        for H in layers:
            h = H[t]
            binder.accumulate_layer(h)
            last_h = h
        assert last_h is not None
        binder.end_token(token=names[t], residual=last_h, entity_name=pos_name.get(t))
    return binder


def _piece(tokenizer: Any, tid: int) -> str:
    return tokenizer.decode([int(tid)], skip_special_tokens=True)


def _last_logits(bundle: Any, hidden: torch.Tensor) -> torch.Tensor:
    from llmintent.forward import get_lm_head, normalize_hidden

    h = normalize_hidden(bundle, hidden.float())
    head = get_lm_head(bundle)
    try:
        return head(h.to(dtype=next(head.parameters()).dtype)).float()
    except Exception:
        from llmintent.models import get_unembedding_matrix

        unembed = get_unembedding_matrix(bundle.model).float()
        return F.linear(h.cpu(), unembed.cpu()).float()


def _top_piece(bundle: Any, logits: torch.Tensor) -> str:
    tid = int(torch.argmax(logits).item())
    return bundle.tokenizer.decode([tid], skip_special_tokens=True).strip() or f"id:{tid}"


def persistence_bind(
    bundle: Any,
    text: str,
    *,
    config: SpikeConfig | None = None,
    gain: float = 1.5,
    inject: bool = True,
) -> BindReport:
    """Forward the transformer, drive the spike net beside it, read live bindings."""
    from llmintent.entity import detect_entity_spans
    from llmintent.forward import forward_hidden_states_from_ids, normalize_hidden

    cfg = config or SpikeConfig()
    ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
    last_str = _piece(bundle.tokenizer, int(ids[0, -1].item()))
    states = forward_hidden_states_from_ids(bundle, ids)
    tokens = [_piece(bundle.tokenizer, int(tid)).strip() for tid in ids[0].tolist()]
    spans = detect_entity_spans(bundle.tokenizer, ids)
    binder = PersistenceBinder(cfg)
    drive_from_hidden(binder, states, tokens=tokens, spans=spans)

    last_h = normalize_hidden(bundle, states[-1][0, -1, :].float())
    mouth_base = _top_piece(bundle, _last_logits(bundle, last_h))
    mouth_bind = mouth_base
    notes = [
        "Spike net is beside the transformer. Decay is not trained.",
        "BoundState locks g once the member set is complete. Last token cannot overwrite it.",
        "Prediction Bayes-updates the mouth with the lock (inform, not +gain).",
        "Trace leak is liveness, not unbinding. This does not claim the base model grew object files.",
    ]
    vec = binder.state_vector()
    if inject and vec is not None:
        from llmintent.predictbind import predict_from_bind

        st = binder.bound_state
        members = list(st.members) if st else []
        mouth_bind = _top_piece(
            bundle,
            predict_from_bind(
                bundle,
                last_h,
                bound_vector=vec,
                members=members,
                mode="bayes",
            ),
        )
        notes.append(
            f"Prediction Bayes-updated from locked state {st.id if st else '?'} "
            f"(locked={st.locked if st else False})."
        )
    live = binder.live_bindings()
    notes.append(
        f"{len(live)} live pair-traces at clock={binder.bank.clock if binder.bank else 0}; "
        f"state locked={binder.bound_state.locked if binder.bound_state else False}."
    )
    return BindReport(
        text=text,
        last_token=last_str,
        decay=cfg.decay,
        trainable_params=0,
        named_traces=binder.named_traces(),
        live_bindings=live,
        expired=list(binder.expired),
        history=binder.history,
        mouth_baseline=mouth_base,
        mouth_with_bindings=mouth_bind,
        bound_scale=float(binder.bound_scale),
        bound_state=binder.bound_state,
        notes=notes,
    )
