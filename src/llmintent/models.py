"""Shared types and model loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


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
        for obj in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)):
            if obj is None:
                continue
            for attr in ("hidden_size", "n_embd", "d_model", "n_embed"):
                v = getattr(obj, attr, None)
                if v:
                    return int(v)
        try:
            layers = get_transformer_layers(self.model)
            mlp = getattr(layers[0], "mlp", None)
            for name in ("down_proj", "c_proj", "dense_4h_to_h"):
                sub = getattr(mlp, name, None) if mlp is not None else None
                w = getattr(sub, "weight", None) if sub is not None else None
                if w is not None and getattr(w, "ndim", 0) == 2:
                    return int(w.shape[0])
        except Exception:
            pass
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
    dtype: str | None = None,
    load_in_4bit: bool = False,
    device_map: str | None = None,
) -> ModelBundle:
    """Load a HuggingFace model for semantic extraction.

    ``load_in_4bit`` uses bitsandbytes NF4 (needed for 27B on a 24 GB GPU).
    When ``device_map`` is set (or 4-bit is on), do not call ``.to(device)``.
    """
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_causal = _is_causal_arch(model_name) if causal is None else causal
    use_device_map = device_map or ("auto" if load_in_4bit else None)

    load_kw: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kw["device_map"] = use_device_map or "auto"
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cap_gb = max(int(props.total_memory / (1024**3)) - 3, 12)
            load_kw["max_memory"] = {0: f"{cap_gb}GiB", "cpu": "64GiB"}
    else:
        dt = _resolve_dtype(dtype)
        load_kw[_dtype_kw()] = dt
        if use_device_map:
            load_kw["device_map"] = use_device_map

    tokenizer = _load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_hf_model(model_name, use_causal=use_causal, load_kw=load_kw)

    if "device_map" not in load_kw:
        model = model.to(resolved_device)
    model.eval()

    try:
        layers = get_transformer_layers(model)
        n_layers = len(layers)
    except AttributeError:
        cfg = getattr(model, "config", None)
        n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        text_cfg = getattr(cfg, "text_config", None)
        if n_layers == 0 and text_cfg is not None:
            n_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)

    param_dev = resolved_device
    try:
        emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if emb is not None and hasattr(emb, "weight"):
            param_dev = emb.weight.device
        else:
            param_dev = next(model.parameters()).device
    except StopIteration:
        pass
    except Exception:
        try:
            param_dev = next(model.parameters()).device
        except StopIteration:
            pass

    return ModelBundle(
        name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=param_dev,
        is_causal=use_causal,
        num_layers=n_layers,
    )


def _dtype_kw() -> str:
    try:
        import transformers

        major = int(str(transformers.__version__).split(".", 1)[0])
        return "dtype" if major >= 5 else "torch_dtype"
    except Exception:
        return "torch_dtype"


def _looks_vl(model_name: str) -> bool:
    n = model_name.lower()
    return any(h in n for h in ("qwen3.8", "qwen3.5", "qwen3_5", "-vl", "vision"))


def _load_tokenizer(model_name: str, *, trust_remote_code: bool):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        return getattr(proc, "tokenizer", proc)


def _load_hf_model(model_name: str, *, use_causal: bool, load_kw: dict[str, Any]):
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM

    if not use_causal:
        try:
            return AutoModelForMaskedLM.from_pretrained(model_name, **load_kw)
        except OSError:
            return AutoModel.from_pretrained(model_name, **load_kw)

    factories = []
    if _looks_vl(model_name):
        try:
            from transformers import AutoModelForImageTextToText

            factories.append(AutoModelForImageTextToText.from_pretrained)
        except Exception:
            pass
    factories.append(AutoModelForCausalLM.from_pretrained)
    try:
        from transformers import AutoModelForImageTextToText as _vl_cls

        if _vl_cls.from_pretrained not in factories:
            factories.append(_vl_cls.from_pretrained)
    except Exception:
        pass
    factories.append(AutoModel.from_pretrained)

    errors: list[str] = []
    for factory in factories:
        try:
            return factory(model_name, **load_kw)
        except Exception as exc:
            errors.append(f"{getattr(factory, '__qualname__', factory)}: {exc}")
    raise RuntimeError(f"Failed to load {model_name!r}: " + " | ".join(errors[-3:]))


def _resolve_dtype(dtype: str | None) -> torch.dtype:
    if dtype is None:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    key = dtype.lower().replace("torch.", "")
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "auto": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    return mapping.get(key, torch.float32)


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
    for path in (
        "model.language_model.layers",
        "model.model.layers",
        "language_model.layers",
        "model.model.language_model.layers",
        "model.language_model.model.layers",
    ):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            try:
                return list(obj)
            except TypeError:
                continue
    raise AttributeError("Unsupported model architecture: cannot locate transformer layers")


def get_ffn_weight(layer: Any) -> torch.Tensor:
    """Extract the primary FFN up-projection weight matrix from a layer.

    Covers GPT-2 ``c_fc``, LLaMA/Qwen ``up_proj``, GLM ``dense_h_to_4h``,
    BERT ``intermediate.dense``, and a few MoE / T5 aliases — any model
    whose blocks expose a feed-forward weight can be anatomized.
    """
    modules = [
        getattr(layer, name, None)
        for name in ("mlp", "ffn", "feed_forward", "feedforward", "moe", "block_sparse_moe")
    ]
    modules = [m for m in modules if m is not None]
    if hasattr(layer, "intermediate"):
        modules.append(layer.intermediate)
    names = (
        "up_proj", "gate_proj", "c_fc", "fc1", "w1", "wi_0", "wi",
        "dense_h_to_4h", "dense", "lin1", "experts",
    )
    for mod in modules:
        for name in names:
            sub = getattr(mod, name, None)
            if sub is None:
                continue
            if name == "experts" and hasattr(sub, "__iter__"):
                try:
                    first = next(iter(sub))
                except TypeError:
                    first = sub[0] if hasattr(sub, "__getitem__") else None
                if first is not None:
                    return get_ffn_weight(first)
            if hasattr(sub, "weight"):
                return sub.weight.data
    if hasattr(layer, "intermediate") and hasattr(layer.intermediate, "dense"):
        return layer.intermediate.dense.weight.data
    raise AttributeError("Layer has no recognized FFN weight")


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
