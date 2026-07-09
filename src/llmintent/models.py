"""Shared types and model loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


@dataclass
class ModelBundle:
    """Loaded tokenizer + model with architecture metadata."""

    name: str
    tokenizer: Any
    model: Any
    device: torch.device
    is_causal: bool
    num_layers: int

    @property
    def hidden_size(self) -> int:
        cfg = self.model.config
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        raise AttributeError(f"Cannot infer hidden size for {self.name}")


class LayerAccessor(Protocol):
    def __iter__(self): ...


def _is_causal_arch(name: str) -> bool:
    lowered = name.lower()
    return any(
        k in lowered
        for k in (
            "gpt",
            "llama",
            "qwen",
            "mistral",
            "ministral",
            "mixtral",
            "phi",
            "minimax",
            "glm",
            "chatglm",
            "opt-",
        )
    )


def load_model_bundle(
    model_name: str,
    *,
    device: str | None = None,
    causal: bool | None = None,
    trust_remote_code: bool = True,
) -> ModelBundle:
    """Load a HuggingFace model for semantic extraction."""
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_causal = _is_causal_arch(model_name) if causal is None else causal

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if use_causal:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,
        )
    else:
        try:
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32,
            )
        except OSError:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32,
            )

    model = model.to(resolved_device)
    model.eval()

    layers = get_transformer_layers(model)
    return ModelBundle(
        name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        is_causal=use_causal,
        num_layers=len(layers),
    )


def get_transformer_layers(model: Any) -> list[Any]:
    """Return the list of transformer blocks for supported architectures."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return list(model.transformer.layer)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return list(model.encoder.layer)
    if hasattr(model, "distilbert") and hasattr(model.distilbert.transformer, "layer"):
        return list(model.distilbert.transformer.layer)
    raise AttributeError("Unsupported model architecture: cannot locate transformer layers")


def get_ffn_weight(layer: Any) -> torch.Tensor:
    """Extract the primary FFN up-projection weight matrix from a layer."""
    if hasattr(layer, "mlp"):
        w_attr = "up_proj" if hasattr(layer.mlp, "up_proj") else "c_fc"
        return getattr(layer.mlp, w_attr).weight.data
    if hasattr(layer, "ffn"):
        return layer.ffn.lin1.weight.data
    if hasattr(layer, "intermediate"):
        return layer.intermediate.dense.weight.data
    raise AttributeError("Layer has no recognized FFN module")


def get_input_embeddings(model: Any) -> torch.Tensor:
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().weight.data
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.weight.data
    if hasattr(model, "distilbert"):
        return model.distilbert.embeddings.word_embeddings.weight.data
    raise AttributeError("Cannot locate input embeddings")


def get_unembedding_matrix(model: Any) -> torch.Tensor:
    """Return unembedding / lm_head weight matrix [vocab_size, hidden_dim]."""
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.data
    if hasattr(model, "cls") and hasattr(model.cls, "predictions"):
        pred = model.cls.predictions
        if hasattr(pred, "decoder") and hasattr(pred.decoder, "weight"):
            return pred.decoder.weight.data
    if hasattr(model, "vocab_projector") and hasattr(model.vocab_projector, "weight"):
        return model.vocab_projector.weight.data
    raise AttributeError("Cannot locate unembedding matrix")
