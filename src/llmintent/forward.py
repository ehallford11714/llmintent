"""Shared forward-pass utilities for hidden-state extraction."""

from __future__ import annotations

from typing import Any

import torch

from llmintent.models import ModelBundle


def forward_hidden_states(bundle: ModelBundle, text: str) -> tuple[Any, list[torch.Tensor]]:
    """Run model and return (inputs, hidden_states) including embedding layer at index 0."""
    inputs = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        if bundle.is_causal:
            if hasattr(bundle.model, "transformer"):
                outputs = bundle.model.transformer(inputs.input_ids, output_hidden_states=True)
            else:
                outputs = bundle.model(**inputs, output_hidden_states=True)
            states = list(outputs.hidden_states)
        else:
            outputs = bundle.model(**inputs, output_hidden_states=True)
            states = list(outputs.hidden_states)
    return inputs, states


def get_lm_head(bundle: ModelBundle) -> torch.nn.Module:
    if hasattr(bundle.model, "lm_head"):
        return bundle.model.lm_head
    if hasattr(bundle.model, "cls"):
        return bundle.model.cls
    if hasattr(bundle.model, "vocab_projector"):
        return bundle.model.vocab_projector
    raise AttributeError("Model has no recognized language modeling head")


def get_final_norm(bundle: ModelBundle) -> torch.nn.Module | None:
    """Return final layer norm before unembedding, if present."""
    model = bundle.model
    candidates = [
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(getattr(model, "model", None), "norm", None),
        getattr(model, "layer_norm", None),
    ]
    for norm in candidates:
        if norm is not None:
            return norm
    return None


def normalize_hidden(bundle: ModelBundle, hidden: torch.Tensor) -> torch.Tensor:
    norm = get_final_norm(bundle)
    if norm is None:
        return hidden
    return norm(hidden)
