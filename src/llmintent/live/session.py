"""Session-scoped model cache for real-time Live inference."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import torch

from llmintent.live.registry import LiveModelSpec, get_live_model
from llmintent.models import ModelBundle, load_model_bundle
from llmintent.retracement.config import RetracementConfig, RetracementMode
from llmintent.retracement.transformer import RetracementTransformer


@dataclass
class LiveSessionConfig:
    """Runtime configuration for a live session."""

    model_key: str = "qwen-0.5b"
    device: str | None = None
    retracement_mode: str = "focus_gate"
    steering_coefficient: float = 0.35
    max_cached_models: int = 1


class LiveSession:
    """
    Keeps one (or few) loaded models hot for real-time analyze / heighten / generate.

    Usage:
        session = LiveSession(LiveSessionConfig(model_key="qwen-0.5b"))
        bundle = session.bundle
        rt = session.retracement
        session.unload()
    """

    def __init__(self, config: LiveSessionConfig | None = None) -> None:
        self.config = config or LiveSessionConfig()
        self._spec = get_live_model(self.config.model_key)
        self._bundle: ModelBundle | None = None
        self._retracement: RetracementTransformer | None = None
        self._cache: dict[str, ModelBundle] = {}

    @property
    def spec(self) -> LiveModelSpec:
        return self._spec

    @property
    def model_key(self) -> str:
        return self._spec.key

    @property
    def bundle(self) -> ModelBundle:
        if self._bundle is None:
            self.load()
        assert self._bundle is not None
        return self._bundle

    @property
    def retracement(self) -> RetracementTransformer:
        if self._retracement is None:
            mode = RetracementMode(self.config.retracement_mode)
            self._retracement = RetracementTransformer(
                self.bundle,
                RetracementConfig(mode=mode),
            )
        return self._retracement

    def load(self, model_key: str | None = None) -> ModelBundle:
        if model_key and model_key != self._spec.key:
            self.switch_model(model_key)
            return self.bundle

        if self._bundle is not None:
            return self._bundle

        key = self._spec.hf_name
        if key in self._cache:
            self._bundle = self._cache[key]
            return self._bundle

        self._bundle = load_model_bundle(
            key,
            device=self.config.device,
            trust_remote_code=True,
        )
        self._cache[key] = self._bundle
        self._trim_cache()
        self._retracement = None
        return self._bundle

    def switch_model(self, model_key: str) -> ModelBundle:
        self._spec = get_live_model(model_key)
        self.config.model_key = model_key
        self._retracement = None
        self._bundle = None
        return self.load()

    def set_retracement_mode(self, mode: str) -> None:
        self.config.retracement_mode = mode
        self._retracement = None

    def _trim_cache(self) -> None:
        if len(self._cache) <= self.config.max_cached_models:
            return
        keep = self._spec.hf_name
        for name in list(self._cache):
            if name != keep:
                del self._cache[name].model
                del self._cache[name]

    def unload(self) -> None:
        self._retracement = None
        self._bundle = None
        for bundle in self._cache.values():
            del bundle.model
        self._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> LiveSession:
        self.load()
        return self

    def __exit__(self, *_args) -> None:
        self.unload()
