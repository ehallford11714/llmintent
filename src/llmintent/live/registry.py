"""Live model registry — Phi-3, Qwen, and SLM targets for real-time use."""

from __future__ import annotations

from llmintent.live.types import LiveModelSpec

LIVE_MODELS: tuple[LiveModelSpec, ...] = (
    LiveModelSpec(
        key="qwen-0.5b",
        hf_name="Qwen/Qwen2.5-0.5B-Instruct",
        params_m=494,
        description="Qwen2.5 0.5B Instruct — fast multilingual SLM",
        chat_template=True,
        default_retracement_mode="focus_gate",
    ),
    LiveModelSpec(
        key="qwen-0.5b-base",
        hf_name="Qwen/Qwen2.5-0.5B",
        params_m=494,
        description="Qwen2.5 0.5B base (completion-style)",
        chat_template=False,
        default_retracement_mode="dual_pass",
    ),
    LiveModelSpec(
        key="phi3-mini",
        hf_name="microsoft/Phi-3-mini-4k-instruct",
        params_m=3800,
        description="Phi-3 Mini 4K Instruct — strong small instruct model",
        chat_template=True,
        default_retracement_mode="focus_gate",
    ),
    LiveModelSpec(
        key="phi2",
        hf_name="microsoft/phi-2",
        params_m=2700,
        description="Phi-2 — compact reasoning SLM",
        chat_template=False,
        default_retracement_mode="dual_pass",
    ),
    LiveModelSpec(
        key="tinyllama",
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        params_m=1100,
        description="TinyLlama 1.1B Chat",
        chat_template=True,
        default_retracement_mode="focus_gate",
    ),
    LiveModelSpec(
        key="gpt2",
        hf_name="gpt2",
        params_m=124,
        description="GPT-2 base — fastest dev / ablation target",
        chat_template=False,
        default_retracement_mode="dual_pass",
    ),
    LiveModelSpec(
        key="distilgpt2",
        hf_name="distilgpt2",
        params_m=82,
        description="DistilGPT-2 — minimal latency baseline",
        chat_template=False,
        default_retracement_mode="extreme",
    ),
)

_REGISTRY: dict[str, LiveModelSpec] = {m.key: m for m in LIVE_MODELS}


def get_live_model(key: str) -> LiveModelSpec:
    if key in _REGISTRY:
        return _REGISTRY[key]
    # Suite keys: "qwen:medium", "mistral:small", …
    if ":" in key and "/" not in key.split(":", 1)[0]:
        try:
            from llmintent.suite import get_model_spec

            fam, _, sz = key.partition(":")
            spec = get_model_spec(fam, sz or "medium")
            return LiveModelSpec(
                key=key,
                hf_name=spec.hf_id,
                params_m=spec.params_b * 1000.0,
                description=spec.description,
                chat_template=spec.chat_template,
            )
        except KeyError:
            pass
    # Allow direct HF hub ids
    return LiveModelSpec(
        key=key,
        hf_name=key,
        params_m=0.0,
        description=f"Custom model: {key}",
        chat_template=False,
    )


def list_live_models() -> list[dict]:
    return [m.to_dict() for m in LIVE_MODELS]


def default_live_model() -> str:
    return "qwen-0.5b"
