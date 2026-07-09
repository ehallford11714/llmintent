"""Resolve suite model IDs from args and environment variables."""

from __future__ import annotations

import os

from llmintent.suite.registry import ModelSpec, get_model_spec

ENV_MODEL = "LLMINTENT_MODEL"
ENV_FAMILY = "LLMINTENT_FAMILY"
ENV_SIZE = "LLMINTENT_SIZE"
ENV_DEVICE = "LLMINTENT_DEVICE"


def resolve_model_spec(
    *,
    model: str | None = None,
    family: str | None = None,
    size: str | None = None,
    use_env: bool = True,
) -> ModelSpec | None:
    """
    Resolve a :class:`ModelSpec` from explicit args and/or env.

    Priority:
      1. Explicit ``model`` HF id / suite key (``qwen:medium``) — returns None
         when ``model`` is a raw HF id (caller should use it directly).
      2. Explicit ``family`` (+ optional ``size``)
      3. Env ``LLMINTENT_FAMILY`` / ``LLMINTENT_SIZE``
      4. Env ``LLMINTENT_MODEL`` as ``family:size`` or raw id → None for raw

    Returns ``None`` when the caller should treat ``model`` / env model as a
    raw Hugging Face id rather than a suite entry.
    """
    if model:
        if ":" in model and not model.startswith(("http://", "https://", "/")):
            fam, _, sz = model.partition(":")
            if fam and sz and "/" not in fam:
                return get_model_spec(fam, sz)
        return None

    fam = family
    sz = size or "medium"
    if use_env:
        fam = fam or os.environ.get(ENV_FAMILY)
        sz = size or os.environ.get(ENV_SIZE) or "medium"
        if not fam:
            env_model = os.environ.get(ENV_MODEL)
            if env_model:
                if ":" in env_model and "/" not in env_model.split(":", 1)[0]:
                    f, _, s = env_model.partition(":")
                    return get_model_spec(f, s or "medium")
                return None

    if fam:
        return get_model_spec(fam, sz)
    return None


def resolve_model_id(
    *,
    model: str | None = None,
    family: str | None = None,
    size: str | None = None,
    use_env: bool = True,
    default: str = "gpt2",
) -> str:
    """
    Resolve a Hugging Face model id string.

    Env vars: ``LLMINTENT_MODEL``, ``LLMINTENT_FAMILY``, ``LLMINTENT_SIZE``.
    """
    if model:
        spec = resolve_model_spec(model=model, use_env=False)
        if spec is not None:
            return spec.hf_id
        return model

    spec = resolve_model_spec(family=family, size=size, use_env=use_env)
    if spec is not None:
        return spec.hf_id

    if use_env:
        env_model = os.environ.get(ENV_MODEL)
        if env_model:
            return env_model

    return default


def resolve_from_env() -> dict[str, str | None]:
    """Return current suite-related environment settings."""
    return {
        "model": os.environ.get(ENV_MODEL),
        "family": os.environ.get(ENV_FAMILY),
        "size": os.environ.get(ENV_SIZE),
        "device": os.environ.get(ENV_DEVICE),
        "resolved_id": resolve_model_id(use_env=True),
    }


def resolve_device(device: str | None = None) -> str | None:
    if device:
        return device
    return os.environ.get(ENV_DEVICE)
