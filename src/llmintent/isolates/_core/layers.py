"""Layer assignment for isolates (abstract L0–L4 + optional HF/llmintent)."""

from __future__ import annotations

import re
from typing import Any, Sequence

from llmintent.isolates._core.types import ABSTRACT_LAYERS, Isolate, TypologyLabel

# Typology → preferred abstract layer
_TYPOLOGY_LAYER: dict[str, int] = {
    TypologyLabel.LEXICAL.value: 0,
    TypologyLabel.NOISE.value: 0,
    TypologyLabel.AFFECTIVE.value: 1,
    TypologyLabel.LATENT_FEATURE.value: 2,
    TypologyLabel.CONFOUNDER.value: 2,
    TypologyLabel.ORPHAN_NODE.value: 2,
    TypologyLabel.UNKNOWN.value: 2,
    TypologyLabel.GOAL.value: 3,
    TypologyLabel.CONSTRAINT.value: 3,
    TypologyLabel.INSTRUMENTAL.value: 3,
    TypologyLabel.ACTION.value: 4,
    TypologyLabel.OUTCOME.value: 4,
}


def layer_name_for(layer: int | str) -> str:
    if isinstance(layer, int) and layer in ABSTRACT_LAYERS:
        return ABSTRACT_LAYERS[layer]
    if isinstance(layer, str):
        return layer
    return f"L{layer}"


def assign_layers(
    isolates: Sequence[Isolate],
    *,
    strategy: str = "abstract",
    layer_map: dict[str, int | str] | None = None,
) -> list[Isolate]:
    """
    Assign ``layer`` / ``layer_name`` to isolates.

    Strategies:
    - ``abstract``: map typology → L0–L4 reasoning layers (offline default)
    - ``span``: early spans → early layers (text position heuristic)
    - ``explicit``: use ``layer_map`` keyed by isolate id
    - ``preserve``: keep existing layer if set, else abstract
    """
    out: list[Isolate] = []
    n = max(len(isolates), 1)
    for i, iso in enumerate(isolates):
        if strategy == "preserve" and iso.layer is not None:
            if not iso.layer_name:
                iso.layer_name = layer_name_for(iso.layer)
            out.append(iso)
            continue

        if strategy == "explicit" and layer_map and iso.id in layer_map:
            iso.layer = layer_map[iso.id]
            iso.layer_name = layer_name_for(iso.layer)
            out.append(iso)
            continue

        if strategy == "span" and iso.span is not None:
            # Quintile of start offset → L0..L4
            start = iso.span[0]
            # Approximate: use index among isolates if no global length
            bucket = min(4, int(5 * i / n))
            if isinstance(start, int) and iso.metadata.get("text_len"):
                tlen = max(int(iso.metadata["text_len"]), 1)
                bucket = min(4, int(5 * start / tlen))
            iso.layer = bucket
            iso.layer_name = ABSTRACT_LAYERS[bucket]
            out.append(iso)
            continue

        # abstract (default)
        typ = iso.typology.value if hasattr(iso.typology, "value") else str(iso.typology)
        layer = _TYPOLOGY_LAYER.get(typ, 2)
        # Slight spread by order within same typology
        iso.layer = layer
        iso.layer_name = ABSTRACT_LAYERS.get(layer, f"L{layer}")
        out.append(iso)
    return list(out)


def soft_llmintent_layers(prompt: str, **kwargs: Any) -> dict[str, Any] | None:
    """
    Optional soft hook: if ``llmintent`` is installed, return a compact
    layer-band summary. Never raises; returns None on failure.
    """
    try:
        from llmintent.layers import summarize_layer_bands  # type: ignore
    except Exception:
        return None
    try:
        # summarize_layer_bands may need a bundle; try import-only metadata path
        return {
            "available": True,
            "hint": "llmintent installed; pass ModelBundle to summarize_layer_bands",
            "prompt_preview": prompt[:80],
            "api": "llmintent.layers.summarize_layer_bands",
            **kwargs,
        }
    except Exception:
        return None


def soft_latentintent_layers(**kwargs: Any) -> dict[str, Any] | None:
    """Optional soft hook for latent thought inspection (external or suite)."""
    try:
        from llmintent import latent as li_latent

        info = li_latent.describe()
        return {
            "available": True,
            "backend": info.get("backend"),
            "external": info.get("external"),
            "hint": "Use llmintent.latent.inspect_text for ThoughtReports",
            **kwargs,
        }
    except Exception:
        pass
    try:
        import latentintent  # type: ignore  # noqa: F401
    except Exception:
        try:
            import latentintentinspect  # type: ignore  # noqa: F401
        except Exception:
            return None
    return {
        "available": True,
        "hint": "latentintent package detected; use llmintent.latent.inspect_text",
        **kwargs,
    }


def parse_layer_token(token: str) -> int | str:
    """Parse '2', 'L2', or 'L2_latent_workspace' into a layer id."""
    token = token.strip()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    m = re.match(r"L(\d+)", token, re.I)
    if m:
        return int(m.group(1))
    return token
