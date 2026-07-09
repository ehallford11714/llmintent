"""SLM registry for benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SLMConfig:
    """Small language model benchmark target."""

    key: str
    hf_name: str
    params_m: float
    description: str


DEFAULT_SLMS: tuple[SLMConfig, ...] = (
    SLMConfig("gpt2", "gpt2", 124, "GPT-2 base (124M)"),
    SLMConfig("distilgpt2", "distilgpt2", 82, "DistilGPT-2 (82M)"),
    SLMConfig("gpt2-medium", "gpt2-medium", 355, "GPT-2 medium (355M)"),
    SLMConfig("opt-125m", "facebook/opt-125m", 125, "OPT-125M"),
)


def get_slm(key: str) -> SLMConfig:
    for slm in DEFAULT_SLMS:
        if slm.key == key:
            return slm
    # Accept suite keys (family:size) and map onto SLMConfig for benchmarks
    if ":" in key and "/" not in key.split(":", 1)[0]:
        try:
            from llmintent.suite import get_model_spec

            fam, _, sz = key.partition(":")
            spec = get_model_spec(fam, sz or "medium")
            return SLMConfig(
                key=key,
                hf_name=spec.hf_id,
                params_m=spec.params_b * 1000.0,
                description=spec.description,
            )
        except KeyError:
            pass
    raise KeyError(f"Unknown SLM {key!r}. Available: {[s.key for s in DEFAULT_SLMS]}")


def list_slms() -> list[dict]:
    return [
        {"key": s.key, "hf_name": s.hf_name, "params_m": s.params_m, "description": s.description}
        for s in DEFAULT_SLMS
    ]
