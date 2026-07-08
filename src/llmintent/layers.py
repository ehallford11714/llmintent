"""Map transformer layers to functional roles and verbal intents."""

from __future__ import annotations

import pandas as pd

from llmintent.activation import identify_activation_layers
from llmintent.cognitive.orchestrator import (
    build_cognitive_module_profile,
    enrich_layer_map_with_cognitive_modules,
)
from llmintent.cognitive.types import CognitiveModuleProfile
from llmintent.jspace.trace import build_intent_trace
from llmintent.jspace.transport import TransportMaps
from llmintent.models import ModelBundle


# Anthropic workspace paper + notebook heuristics
LAYER_ROLE_HINTS: dict[str, str] = {
    "sensory": "Token ingestion & local feature binding",
    "workspace": "Abstract reasoning & silent verbal thoughts (J-space)",
    "motor": "Next-token readout alignment & formatting",
}


def build_layer_correspondence_map(
    bundle: ModelBundle,
    prompt: str,
    *,
    transport: TransportMaps | None = None,
    position: int = -1,
    twin_b: str | None = None,
    cognitive_profile: CognitiveModuleProfile | None = None,
) -> pd.DataFrame:
    """
    Generate a layer-by-layer correspondence table for a transformer.

    Columns: layer, normalized_depth, regime, role, top_intent, occupancy,
    entropy, motor_alignment, is_activation_pivot
    """
    trace = build_intent_trace(
        bundle,
        prompt,
        transport=transport,
        position=position,
    )
    activation = identify_activation_layers(
        bundle,
        prompt,
        layer_stats=trace.layer_stats,
        entropy=trace.entropy,
        occupancy=trace.occupancy,
        position=position,
    )
    pivot_layers = set(activation.values())

    rows: list[dict] = []
    for _, row in trace.layer_stats.iterrows():
        layer = int(row["layer"])
        regime = row["regime"]
        rows.append(
            {
                "layer": layer,
                "normalized_depth": float(row["normalized_depth"]),
                "regime": regime,
                "role": LAYER_ROLE_HINTS.get(regime, "Intermediate processing"),
                "top_intent": row.get("top1_intent", ""),
                "top_intent_prob": float(row.get("top1_prob", 0.0)),
                "occupancy": float(row.get("occupancy", 0.0)),
                "entropy": float(row.get("entropy", 0.0)),
                "motor_alignment": float(row.get("motor_alignment", 0.0)),
                "interpretability": float(row.get("interpretability", 0.0)),
                "is_activation_pivot": layer in pivot_layers,
                "activation_tags": _activation_tags(layer, activation),
            }
        )
    df = pd.DataFrame(rows)

    if cognitive_profile is None and twin_b is not None:
        cognitive_profile = build_cognitive_module_profile(
            bundle,
            prompt,
            twin_b,
            transport=transport,
            position=position,
        )
    if cognitive_profile is not None:
        df = enrich_layer_map_with_cognitive_modules(df, cognitive_profile)
    return df


def summarize_layer_bands(
    bundle: ModelBundle,
    prompt: str,
    *,
    transport: TransportMaps | None = None,
) -> dict:
    """High-level summary of which layer ranges correspond to which function."""
    trace = build_intent_trace(bundle, prompt, transport=transport)
    bands = trace.regime_bands
    activation = trace.activation_layers
    return {
        "model": bundle.name,
        "prompt": prompt,
        "num_layers": trace.num_layers,
        "regime_bands": bands,
        "role_hints": LAYER_ROLE_HINTS,
        "activation_layers": activation,
        "workspace_thoughts": [
            trace.layer_stats.loc[i, "top1_intent"]
            for i in trace.workspace_layers()
            if i < len(trace.layer_stats)
        ][:5],
    }


def compare_logit_vs_j_lens(
    bundle: ModelBundle,
    prompt: str,
    transport: TransportMaps,
    *,
    position: int = -1,
) -> pd.DataFrame:
    """Compare logit lens vs transport (J-lens proxy) decode at each layer."""
    from llmintent.jspace.decode import decode_intents

    trace = build_intent_trace(bundle, prompt, transport=transport, position=position)
    _, states = __import__("llmintent.forward", fromlist=["forward_hidden_states"]).forward_hidden_states(
        bundle, prompt
    )
    pos = position if position >= 0 else trace.seq_len + position

    rows: list[dict] = []
    for layer_idx, state in enumerate(states):
        hidden = state[0, pos, :]
        logit = decode_intents(bundle, hidden, layer=layer_idx, transport=None, top_k=1)
        jlens = decode_intents(
            bundle,
            hidden,
            layer=layer_idx,
            transport=transport.get(layer_idx),
            top_k=1,
        )
        rows.append(
            {
                "layer": layer_idx,
                "logit_lens_top": logit[0].token if logit else "",
                "j_lens_top": jlens[0].token if jlens else "",
                "same_intent": (logit and jlens and logit[0].token_id == jlens[0].token_id),
            }
        )
    return pd.DataFrame(rows)


def _activation_tags(layer: int, activation: dict[str, int]) -> str:
    tags = [name for name, idx in activation.items() if idx == layer]
    return ", ".join(tags) if tags else ""
