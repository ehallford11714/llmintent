"""ThoughtReport builder (vendored offline path)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from llmintent.latent_vendor.probes import LinearIntentProbe, score_vector_with_probe
from llmintent.latent_vendor.rule_backend import heuristic_layer_saliency, rule_label_text
from llmintent.latent_vendor.sae_lite import SAELite
from llmintent.latent_vendor.types import EPISTEMIC_CAVEATS, IntentHypothesis, LayerSaliency

_VENDOR_VERSION = "0.1.0"


@dataclass
class ThoughtReport:
    text: str
    backend: str
    hypothesized_intents: list[IntentHypothesis] = field(default_factory=list)
    layer_saliency: list[LayerSaliency] = field(default_factory=list)
    logit_lens: list[dict[str, Any]] = field(default_factory=list)
    sae_features: dict[str, Any] | None = None
    probe_metrics: dict[str, Any] | None = None
    model_name: str | None = None
    caveats: list[str] = field(default_factory=lambda: list(EPISTEMIC_CAVEATS))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "llmintent.latent.ThoughtReport/v0.1",
            "version": _VENDOR_VERSION,
            "created_at": self.created_at,
            "text": self.text,
            "backend": self.backend,
            "model_name": self.model_name,
            "hypothesized_intents": [h.to_dict() for h in self.hypothesized_intents],
            "layer_saliency": [s.to_dict() for s in self.layer_saliency],
            "logit_lens": self.logit_lens,
            "sae_features": self.sae_features,
            "probe_metrics": self.probe_metrics,
            "caveats": self.caveats,
            "metadata": self.metadata,
            "disclaimer": (
                "This report does not claim to read true model or human thoughts. "
                "It summarizes probes, heuristics, and optional activation tools."
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_lines(self) -> list[str]:
        lines = [
            f"backend={self.backend}",
            f"intents={', '.join(f'{h.tag}:{h.score:.2f}' for h in self.hypothesized_intents[:5])}",
        ]
        if self.layer_saliency:
            top = max(self.layer_saliency, key=lambda s: s.score)
            lines.append(f"top_layer_saliency=L{top.layer}:{top.score:.2f} ({top.source})")
        lines.append("caveat: correlates/probes only — not mind-reading")
        return lines


def build_thought_report(
    text: str,
    *,
    backend: str = "rule",
    model: str | None = None,
    family: str | None = None,
    include_sae: bool = True,
    include_probe_train: bool = True,
    **_kwargs: Any,
) -> ThoughtReport:
    """
    Build a ThoughtReport.

    Vendored path supports ``backend=rule`` offline. For HF capture, install
    the extractable ``latentintent`` package (``pip install latentintent`` or
    ``llmintent[latent]``) which soft-overrides this module.
    """
    backend = (backend or "rule").lower().strip()
    meta_note: str | None = None
    if backend not in ("rule", "heuristic"):
        # Soft: try external package for HF; else fall back to rule with note
        try:
            import latentintent as ext  # type: ignore

            return ext.inspect_text(
                text,
                backend=backend,
                model=model,
                family=family,
                include_sae=include_sae,
                include_probe_train=include_probe_train,
            )
        except Exception as exc:
            meta_note = f"hf_unavailable_fallback_rule: {exc}"
            backend = "rule"

    hyps = rule_label_text(text)
    sal = heuristic_layer_saliency(text)
    probe_metrics = None
    sae_features = None
    logit: list[dict[str, Any]] = []

    if include_probe_train:
        probe = LinearIntentProbe(random_state=0)
        _, pres = probe.fit_synthetic(n_samples=160, dim=32)
        probe_metrics = {
            "accuracy": pres.accuracy,
            "auroc": pres.auroc,
            "synthetic": True,
            "note": "Synthetic planted-direction eval — not measured on model activations.",
        }
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.normal(size=32)
        for h in score_vector_with_probe(probe, vec).top_hypotheses(3):
            hyps.append(h)

    if include_sae:
        sae = SAELite(n_components=12, random_state=1)
        sae_features = sae.fit_encode_synthetic(dim=32, n_samples=48).to_dict()

    # Tiny logit-lens stub (numpy only)
    rng = np.random.default_rng(42)
    h = rng.normal(size=32)
    W = rng.normal(size=(40, 32))
    logits = W @ h
    top_idx = np.argsort(logits)[::-1][:5]
    logit = [
        {
            "layer": 2,
            "top_tokens": [{"token": f"tok_{int(i)}", "logit": float(logits[int(i)])} for i in top_idx],
            "method": "logit_lens_stub",
            "metadata": {"note": "Offline stub; install latentintent[hf] for real unembed."},
        }
    ]

    meta: dict[str, Any] = {"mode": "offline_rule", "source": "llmintent.latent_vendor"}
    if meta_note:
        meta["note"] = meta_note
    if model or family:
        meta["requested_model"] = model
        meta["requested_family"] = family

    return ThoughtReport(
        text=text,
        backend="rule",
        hypothesized_intents=hyps,
        layer_saliency=sal,
        logit_lens=logit,
        sae_features=sae_features,
        probe_metrics=probe_metrics,
        model_name=None,
        metadata=meta,
    )


def inspect_text(text: str, **kwargs: Any) -> ThoughtReport:
    return build_thought_report(text, **kwargs)
