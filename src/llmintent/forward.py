"""Shared forward-pass utilities for hidden-state extraction."""

from __future__ import annotations

from typing import Any

import torch

from llmintent.models import ModelBundle


def encode_prompt_ids(bundle: ModelBundle, text: str, *, thinking: bool = False) -> torch.Tensor:
    """Tokenize ``text``; disable Qwen thinking when the chat template supports it."""
    tok = bundle.tokenizer
    messages = [{"role": "user", "content": text}]
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            ids = tok.apply_chat_template(
                messages,
                enable_thinking=thinking,
                **kwargs,
            )
        except TypeError:
            try:
                ids = tok.apply_chat_template(
                    messages,
                    chat_template_kwargs={"enable_thinking": thinking},
                    **kwargs,
                )
            except TypeError:
                ids = tok.apply_chat_template(messages, **kwargs)
        if isinstance(ids, torch.Tensor):
            return ids.to(bundle.device)
        if isinstance(ids, dict) and "input_ids" in ids:
            return ids["input_ids"].to(bundle.device)
    inputs = tok(text, return_tensors="pt")
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    return ids.to(bundle.device)


def forward_hidden_states(bundle: ModelBundle, text: str) -> tuple[Any, list[torch.Tensor]]:
    """Run model and return (inputs, hidden_states) including embedding layer at index 0."""
    ids = encode_prompt_ids(bundle, text, thinking=False)
    states = forward_hidden_states_from_ids(bundle, ids)
    return {"input_ids": ids}, states


def forward_hidden_states_from_ids(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Extract hidden states for pre-tokenized input ids."""
    if input_ids.device != bundle.device:
        input_ids = input_ids.to(bundle.device)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        if bundle.is_causal:
            if hasattr(bundle.model, "transformer") and not hasattr(bundle.model, "language_model"):
                outputs = bundle.model.transformer(input_ids, output_hidden_states=True)
            else:
                try:
                    outputs = bundle.model(
                        input_ids,
                        attention_mask=attn,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                except TypeError:
                    outputs = bundle.model(
                        input_ids,
                        output_hidden_states=True,
                    )
            hidden = getattr(outputs, "hidden_states", None)
            if hidden is None:
                raise RuntimeError("Model forward did not return hidden_states")
            return list(hidden)
        outputs = bundle.model(input_ids, output_hidden_states=True)
        return list(outputs.hidden_states)


def get_lm_head(bundle: ModelBundle) -> torch.nn.Module:
    model = bundle.model
    for path in (
        "lm_head",
        "language_model.lm_head",
        "model.lm_head",
        "model.language_model.lm_head",
        "cls",
        "vocab_projector",
    ):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and obj is not None:
            return obj
    raise AttributeError("Model has no recognized language modeling head")


def get_final_norm(bundle: ModelBundle) -> torch.nn.Module | None:
    """Return final layer norm before unembedding, if present."""
    model = bundle.model
    candidates = [
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(getattr(model, "model", None), "norm", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "norm", None),
        getattr(getattr(model, "language_model", None), "norm", None),
        getattr(getattr(getattr(model, "language_model", None), "model", None), "norm", None),
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
    try:
        return norm(hidden)
    except Exception:
        return hidden
