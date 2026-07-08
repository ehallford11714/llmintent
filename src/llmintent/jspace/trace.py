"""Build layer × position intent traces (global workspace view)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch
import torch.nn.functional as F

from llmintent.forward import forward_hidden_states, normalize_hidden
from llmintent.jspace.decode import decode_intents
from llmintent.jspace.decompose import jspace_occupancy
from llmintent.jspace.regimes import classify_layer_regimes, regime_bands
from llmintent.jspace.transport import TransportMaps
from llmintent.metrics import shannon_entropy
from llmintent.models import ModelBundle, get_unembedding_matrix


@dataclass
class IntentTrace:
    """Layer × position verbal intent trajectory for a prompt."""

    prompt: str
    model_name: str
    num_layers: int
    seq_len: int
    top1_grid: list[list[str]]
    top1_ids: list[list[int]]
    rank_curves: dict[str, list[int | None]] = field(default_factory=dict)
    occupancy: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    layer_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_bands: dict[str, tuple[int, int]] = field(default_factory=dict)
    activation_layers: dict[str, int] = field(default_factory=dict)

    def top_thought_at(self, layer: int, position: int = -1) -> str:
        pos = position if position >= 0 else self.seq_len + position
        return self.top1_grid[layer][pos]

    def workspace_layers(self) -> range:
        band = self.regime_bands.get("workspace", (0, self.num_layers))
        return range(band[0], band[1] + 1)


def build_intent_trace(
    bundle: ModelBundle,
    prompt: str,
    *,
    transport: TransportMaps | None = None,
    track_tokens: list[str] | None = None,
    top_k: int = 5,
    position: int = -1,
) -> IntentTrace:
    """Run forward pass and decode verbal intents at every layer."""
    _, states = forward_hidden_states(bundle, prompt)
    seq_len = states[0].shape[1]
    pos = position if position >= 0 else seq_len + position
    num_layers = len(states)

    top1_grid: list[list[str]] = []
    top1_ids: list[list[int]] = []
    occupancy: list[float] = []
    entropy: list[float] = []
    top1_tokens: list[str] = []
    motor_logits: torch.Tensor | None = None

    unembed = get_unembedding_matrix(bundle.model)
    final_h = states[-1][0, pos, :]
    motor_logits = F.linear(normalize_hidden(bundle, final_h), unembed)

    rows: list[dict] = []

    for layer_idx, state in enumerate(states):
        hidden = state[0, pos, :]
        t_map = transport.get(layer_idx) if transport else None
        intents = decode_intents(bundle, hidden, layer=layer_idx, transport=t_map, top_k=top_k)

        row_tokens = []
        row_ids = []
        for p in range(seq_len):
            h_p = state[0, p, :]
            top = decode_intents(bundle, h_p, layer=layer_idx, transport=t_map, top_k=1)
            row_tokens.append(top[0].token if top else "")
            row_ids.append(top[0].token_id if top else -1)
        top1_grid.append(row_tokens)
        top1_ids.append(row_ids)

        occ = jspace_occupancy(bundle, hidden, transport=t_map)
        occupancy.append(float(occ))

        logits = F.linear(normalize_hidden(bundle, hidden if t_map is None else hidden @ t_map.to(hidden.device).T), unembed)
        ent = shannon_entropy(F.softmax(logits, dim=-1))
        entropy.append(ent)

        top1 = intents[0].token if intents else ""
        top1_tokens.append(top1)

        # Motor alignment: cosine similarity of layer decode to final layer decode
        layer_probs = F.softmax(logits, dim=-1)
        motor_probs = F.softmax(motor_logits, dim=-1)
        motor_align = float(F.cosine_similarity(layer_probs.unsqueeze(0), motor_probs.unsqueeze(0)).item())

        rows.append(
            {
                "layer": layer_idx,
                "top1_intent": top1,
                "top1_prob": intents[0].probability if intents else 0.0,
                "occupancy": occ,
                "entropy": ent,
                "motor_alignment": motor_align,
            }
        )

    # Top-1 stability: how often top intent matches final layer
    final_top = top1_tokens[-1]
    stability = [1.0 if t == final_top else 0.0 for t in top1_tokens]
    for i, row in enumerate(rows):
        row["top1_stability"] = stability[i]
        row["interpretability"] = 1.0 - (entropy[i] / (max(entropy) + 1e-8))

    layer_stats = classify_layer_regimes(pd.DataFrame(rows), num_layers=bundle.num_layers)
    bands = regime_bands(layer_stats)

    rank_curves: dict[str, list[int | None]] = {}
    if track_tokens:
        for token in track_tokens:
            token_ids = bundle.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids:
                continue
            tid = token_ids[0]
            ranks: list[int | None] = []
            for layer_idx, state in enumerate(states):
                hidden = state[0, pos, :]
                t_map = transport.get(layer_idx) if transport else None
                intents = decode_intents(bundle, hidden, layer=layer_idx, transport=t_map, top_k=100)
                rank_val = next((it.rank for it in intents if it.token_id == tid), None)
                ranks.append(rank_val)
            rank_curves[token] = ranks

    from llmintent.activation import identify_activation_layers

    activation = identify_activation_layers(
        bundle,
        prompt,
        layer_stats=layer_stats,
        entropy=entropy,
        occupancy=occupancy,
        position=pos,
    )

    return IntentTrace(
        prompt=prompt,
        model_name=bundle.name,
        num_layers=num_layers,
        seq_len=seq_len,
        top1_grid=top1_grid,
        top1_ids=top1_ids,
        rank_curves=rank_curves,
        occupancy=occupancy,
        entropy=entropy,
        layer_stats=layer_stats,
        regime_bands={
            "sensory": bands.sensory,
            "workspace": bands.workspace,
            "motor": bands.motor,
        },
        activation_layers=activation,
    )
