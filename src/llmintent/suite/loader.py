"""Lazy Hugging Face loaders for the model suite (no import-time downloads)."""

from __future__ import annotations

import os
from typing import Any

from llmintent.suite.registry import ModelSpec, get_model_spec
from llmintent.suite.resolve import resolve_device, resolve_model_id, resolve_model_spec


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required to load suite models. "
            'Install with: pip install "llmintent[models]"'
        ) from exc


def load_suite_model(
    family: str | None = None,
    size: str = "medium",
    *,
    model: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    trust_remote_code: bool | None = None,
) -> Any:
    """
    Soft-load a suite model via :func:`llmintent.models.load_model_bundle`.

    Never downloads at import time. Heavy deps (torch / transformers) are
    imported only when this function is called.
    """
    from llmintent.models import load_model_bundle

    _require_transformers()

    spec = resolve_model_spec(model=model, family=family, size=size)
    if spec is not None:
        hf_id = spec.hf_id
        trc = trust_remote_code if trust_remote_code is not None else spec.trust_remote_code
    else:
        hf_id = resolve_model_id(model=model, family=family, size=size)
        trc = True if trust_remote_code is None else trust_remote_code

    resolved_device = resolve_device(device)
    # dtype reserved for future float16/bfloat16 path; load_model_bundle uses float32 today
    _ = dtype
    return load_model_bundle(hf_id, device=resolved_device, trust_remote_code=trc)


def soft_pipeline(
    family: str | None = None,
    size: str = "medium",
    *,
    model: str | None = None,
    task: str = "text-generation",
    device: str | None = None,
    **pipeline_kwargs: Any,
) -> Any:
    """
    Lazily construct a ``transformers.pipeline`` for a suite model.

    Raises ImportError if transformers is missing. Does not download until
    the pipeline factory runs.
    """
    _require_transformers()
    from transformers import pipeline

    spec = resolve_model_spec(model=model, family=family, size=size)
    if spec is not None:
        model_id = spec.hf_id
        trc = spec.trust_remote_code
    else:
        model_id = resolve_model_id(model=model, family=family, size=size)
        trc = True

    resolved = resolve_device(device)
    kwargs = dict(pipeline_kwargs)
    kwargs.setdefault("trust_remote_code", trc)
    if resolved:
        # pipeline device: -1 cpu, or cuda index
        if resolved == "cpu":
            kwargs.setdefault("device", -1)
        elif resolved.startswith("cuda"):
            idx = 0
            if ":" in resolved:
                try:
                    idx = int(resolved.split(":", 1)[1])
                except ValueError:
                    idx = 0
            kwargs.setdefault("device", idx)

    return pipeline(task, model=model_id, **kwargs)


def should_run_load_tests() -> bool:
    """Gate multi-GB download tests behind ``LLMINTENT_LOAD_TEST=1``."""
    return os.environ.get("LLMINTENT_LOAD_TEST", "").strip() in {"1", "true", "yes"}


def spec_or_raise(family: str, size: str) -> ModelSpec:
    return get_model_spec(family, size)
