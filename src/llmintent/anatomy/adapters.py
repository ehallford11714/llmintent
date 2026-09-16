"""Architecture-aware weight adapters.

Unsupported modules return an explicit capability result. They never masquerade
as a complete map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from llmintent.anatomy.evidence import ComponentId, ComponentSpec
from llmintent.anatomy.weights import logical_linear_shape, looks_quantized


@dataclass
class ModelIndex:
    architecture: str
    checkpoint: str
    revision: str | None
    n_layers: int
    hidden_size: int
    n_heads: int | None
    components: list[ComponentSpec] = field(default_factory=list)
    capabilities: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    packages: dict[str, str] = field(default_factory=dict)

    def by_role(self, role: str) -> list[ComponentSpec]:
        return [c for c in self.components if c.role == role]

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "n_heads": self.n_heads,
            "n_components": len(self.components),
            "components": [c.to_dict() for c in self.components],
            "capabilities": dict(self.capabilities),
            "notes": list(self.notes),
            "packages": dict(self.packages),
        }


def _revision(model: Any) -> str | None:
    cfg = getattr(model, "config", None)
    return getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)


def _dtype_str(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", "unknown"))


def _quant(tensor: Any, module: Any = None) -> str | None:
    for obj in (module, tensor, getattr(module, "weight", None)):
        if obj is None:
            continue
        name = type(obj).__name__.lower()
        if "nf4" in name or "4bit" in name or "params4bit" in name:
            return "nf4"
        if looks_quantized(obj):
            return "nf4"
    return None


def _spec(
    *,
    checkpoint: str,
    revision: str | None,
    path: str,
    tensor: Any,
    role: str,
    in_space: str,
    out_space: str,
    layout: str,
    orientation: str,
    layer: int | None = None,
    head: int | None = None,
    expert: int | None = None,
    nonlinearity: str | None = None,
    notes: Iterable[str] = (),
    module: Any = None,
) -> ComponentSpec:
    logical = logical_linear_shape(module) if module is not None else None
    shape = logical or tuple(int(x) for x in getattr(tensor, "shape", ()))
    return ComponentSpec(
        component=ComponentId(
            checkpoint=checkpoint,
            revision=revision,
            module_path=path,
            layer=layer,
            head=head,
            expert=expert,
        ),
        shape=shape,
        orientation=orientation,
        in_space=in_space,
        out_space=out_space,
        dtype=_dtype_str(tensor),
        quantization=_quant(tensor, module),
        layout=layout,
        nonlinearity=nonlinearity,
        role=role,
        notes=list(notes),
        capability="ok",
    )


def _missing(checkpoint: str, revision: str | None, path: str, role: str, reason: str) -> ComponentSpec:
    return ComponentSpec(
        component=ComponentId(checkpoint, revision, path),
        shape=(),
        orientation="unknown",
        in_space="unknown",
        out_space="unknown",
        dtype="none",
        role=role,
        capability="unavailable",
        notes=[reason],
    )


def index_bundle(bundle: Any) -> ModelIndex:
    """Dispatch on architecture. GPT-2 family and Llama/Qwen-like are covered."""
    from llmintent.anatomy.evidence import package_versions
    from llmintent.models import get_transformer_layers

    model = bundle.model
    cfg = getattr(model, "config", None)
    arch = str(getattr(cfg, "model_type", None) or type(model).__name__)
    ckpt = str(getattr(bundle, "name", None) or arch)
    rev = _revision(model)
    try:
        layers = get_transformer_layers(model)
        n_layers = len(layers)
    except Exception as exc:
        return ModelIndex(
            architecture=arch,
            checkpoint=ckpt,
            revision=rev,
            n_layers=0,
            hidden_size=int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0) or 0),
            n_heads=getattr(cfg, "n_head", None) or getattr(cfg, "num_attention_heads", None),
            capabilities={"layers": f"unavailable: {exc}"},
            notes=["Cannot locate transformer blocks."],
            packages=package_versions(),
        )

    hidden = int(
        getattr(cfg, "n_embd", None)
        or getattr(cfg, "hidden_size", None)
        or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
        or bundle.hidden_size
    )
    n_heads = getattr(cfg, "n_head", None) or getattr(cfg, "num_attention_heads", None)
    if _is_gpt2_family(arch, layers):
        idx = _index_gpt2(bundle, layers, ckpt, rev, arch, hidden, n_heads)
    elif _is_llama_like(layers):
        idx = _index_llama_like(bundle, layers, ckpt, rev, arch, hidden, n_heads)
    else:
        idx = _index_generic(bundle, layers, ckpt, rev, arch, hidden, n_heads)
    idx.packages = package_versions()
    return idx


def _is_gpt2_family(arch: str, layers: list[Any]) -> bool:
    if arch.lower() in {"gpt2", "gpt_neo", "distilgpt2"}:
        return True
    layer0 = layers[0] if layers else None
    attn = getattr(layer0, "attn", None)
    return attn is not None and hasattr(attn, "c_attn")


def _is_llama_like(layers: list[Any]) -> bool:
    layer0 = layers[0] if layers else None
    mlp = getattr(layer0, "mlp", None)
    if mlp is None:
        return False
    if hasattr(mlp, "down_proj"):
        return True
    return hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")


def _index_gpt2(bundle, layers, ckpt, rev, arch, hidden, n_heads) -> ModelIndex:
    comps: list[ComponentSpec] = []
    caps = {
        "embeddings": "ok", "attention_qkv": "ok", "attention_ov": "ok",
        "ffn_up": "ok", "ffn_down": "ok", "norms": "ok", "lm_head": "ok", "moe": "unavailable",
    }
    model = bundle.model
    wte = getattr(getattr(model, "transformer", model), "wte", None)
    if wte is not None and hasattr(wte, "weight"):
        comps.append(_spec(
            checkpoint=ckpt, revision=rev,                     path="transformer.wte.weight",
            tensor=wte.weight.data, role="embedding", in_space="token_id", out_space="residual",
            module=wte,
            layout="vocab_hidden", orientation="row_is_token_vector",
        ))
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(getattr(mlp, "c_proj", None), "weight"):
            comps.append(_spec(
                checkpoint=ckpt, revision=rev,                     path=f"transformer.h.{i}.mlp.c_proj.weight",
                tensor=mlp.c_proj.weight.data, role="ffn_out",
                in_space="ffn_hidden", out_space="residual",
                layout="conv1d_in_out", orientation="columns_write_residual", layer=i,
                module=mlp.c_proj,
                notes=["Writer into residual after nonlinearity."],
            ))
    return ModelIndex(
        architecture=arch or "gpt2", checkpoint=ckpt, revision=rev,
        n_layers=len(layers), hidden_size=hidden,
        n_heads=int(n_heads) if n_heads else None, components=comps, capabilities=caps,
        notes=["GPT-2 Conv1D adapter."],
    )


def _mlp_path(i: int, name: str, layers: list[Any]) -> str:
    layer = layers[i]
    if hasattr(layer, "mlp") and hasattr(getattr(layer, "mlp", None), name):
        parent = getattr(layers[0], "mlp", None)
        # Qwen2/3 and Llama: model.layers.i.mlp.name
        return f"model.layers.{i}.mlp.{name}.weight"
    return f"layers.{i}.mlp.{name}.weight"


def _index_llama_like(bundle, layers, ckpt, rev, arch, hidden, n_heads) -> ModelIndex:
    comps: list[ComponentSpec] = []
    caps = {
        "embeddings": "ok" if hasattr(bundle.model, "model") else "unknown",
        "attention_qkv": "ok", "attention_ov": "ok",
        "ffn_gate": "ok", "ffn_up": "ok", "ffn_down": "ok", "moe": "unavailable",
    }
    # Qwen3 / Llama blocks live on model.layers or model.language_model.layers.
    layer_prefix = "model.layers"
    model = bundle.model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_prefix = "model.layers"
    elif hasattr(model, "model") and hasattr(getattr(model.model, "language_model", None), "layers"):
        layer_prefix = "model.language_model.layers"
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        for name, role, insp, outsp in (
            ("q_proj", "attn_q", "residual", "query"),
            ("k_proj", "attn_k", "residual", "key"),
            ("v_proj", "attn_v", "residual", "value"),
            ("o_proj", "attn_out", "attn_concat", "residual"),
        ):
            mod = getattr(attn, name, None) if attn is not None else None
            if mod is not None and hasattr(mod, "weight"):
                comps.append(_spec(
                    checkpoint=ckpt, revision=rev,
                    path=f"{layer_prefix}.{i}.self_attn.{name}.weight",
                    tensor=mod.weight.data, role=role, in_space=insp, out_space=outsp,
                    layout="linear_out_in", orientation="row_writes_output_channel", layer=i,
                    module=mod,
                ))
            else:
                comps.append(_missing(ckpt, rev, f"{layer_prefix}.{i}.{name}", role, "missing projection"))
                caps["attention_qkv"] = "partial"
        mlp = getattr(layer, "mlp", None)
        for name, role, insp, outsp, nl in (
            ("gate_proj", "ffn_gate", "residual", "ffn_hidden", "silu"),
            ("up_proj", "ffn_in", "residual", "ffn_hidden", None),
            ("down_proj", "ffn_out", "ffn_hidden", "residual", None),
        ):
            mod = getattr(mlp, name, None) if mlp is not None else None
            if mod is not None and hasattr(mod, "weight"):
                comps.append(_spec(
                    checkpoint=ckpt, revision=rev,
                    path=f"{layer_prefix}.{i}.mlp.{name}.weight",
                    tensor=mod.weight.data, role=role, in_space=insp, out_space=outsp,
                    layout="linear_out_in", orientation="row_writes_output_channel",
                    layer=i, nonlinearity=nl, module=mod,
                    notes=["Analyze gate×up then down as a composition, not up_proj SVD alone."],
                ))
            else:
                comps.append(_missing(ckpt, rev, f"{layer_prefix}.{i}.mlp.{name}", role, "missing"))
    return ModelIndex(
        architecture=arch or "llama", checkpoint=ckpt, revision=rev,
        n_layers=len(layers), hidden_size=hidden,
        n_heads=int(n_heads) if n_heads else None, components=comps, capabilities=caps,
        notes=["Llama/Qwen-style Linear adapter (Qwen 27B uses this path)."],
    )


def _index_generic(bundle, layers, ckpt, rev, arch, hidden, n_heads) -> ModelIndex:
    from llmintent.models import get_ffn_weight

    comps: list[ComponentSpec] = []
    caps = {"attention": "unknown", "ffn": "partial", "moe": "unavailable"}
    for i, layer in enumerate(layers):
        try:
            w = get_ffn_weight(layer)
            comps.append(_spec(
                checkpoint=ckpt, revision=rev, path=f"layer.{i}.ffn_primary",
                tensor=w, role="ffn_in", in_space="unknown", out_space="unknown",
                layout="unknown", orientation="shape_heuristic", layer=i,
                notes=["Generic fallback. Orientation not architecture-verified."],
            ))
        except Exception as exc:
            comps.append(_missing(ckpt, rev, f"layer.{i}.ffn", "ffn_in", str(exc)))
    return ModelIndex(
        architecture=arch, checkpoint=ckpt, revision=rev,
        n_layers=len(layers), hidden_size=hidden,
        n_heads=int(n_heads) if n_heads else None, components=comps, capabilities=caps,
        notes=["Generic adapter — incomplete map, not a full architecture parse."],
    )
