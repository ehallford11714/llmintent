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
    if backend in ("hf", "residual", "anatomy"):
        try:
            return _inspect_hf_residuals(
                text,
                model=model,
                family=family,
                include_sae=include_sae,
                include_probe_train=include_probe_train,
                **_kwargs,
            )
        except Exception as exc:
            # Do not silently stub a 27B / explicit HF request.
            mid = str(model or family or "")
            if _kwargs.get("require_hf") or "27" in mid.lower() or "qwen3.8" in mid.lower():
                raise
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
            except Exception as ext_exc:
                meta_note = f"hf_unavailable_fallback_rule: {exc}; latentintent={ext_exc}"
                backend = "rule"

    if backend not in ("rule", "heuristic"):
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


def _inspect_hf_residuals(
    text: str,
    *,
    model: str | None,
    family: str | None,
    include_sae: bool,
    include_probe_train: bool,
    **kwargs: Any,
) -> ThoughtReport:
    from llmintent.anatomy.thoughts import inspect_latent_thoughts
    from llmintent.models import load_model_bundle
    from llmintent.suite import resolve_model_spec
    from llmintent.suite.resolve import resolve_model_id

    size = kwargs.get("size")
    fourbit = kwargs.get("load_in_4bit")
    spec = None
    if model:
        spec = resolve_model_spec(model=model, use_env=False)
    elif family:
        spec = resolve_model_spec(family=family, size=size or "medium", use_env=False)
    else:
        spec = resolve_model_spec(model="qwen:27b", use_env=False)

    if spec is not None:
        model_id = spec.hf_id
        candidates = (spec.hf_id,) + tuple(spec.alternates)
        if fourbit is None and spec.size == "27b":
            fourbit = True
    else:
        model_id = resolve_model_id(model=model, family=family, size=size, default="gpt2")
        candidates = (model_id,)
        if fourbit is None and "27B" in str(model_id):
            fourbit = True

    last_exc: Exception | None = None
    for mid in candidates:
        if not mid or ("/" not in str(mid) and ":" in str(mid)):
            continue
        if fourbit and "fp8" in str(mid).lower():
            continue
        try:
            bundle = load_model_bundle(str(mid), load_in_4bit=bool(fourbit))
            lat = inspect_latent_thoughts(
                bundle,
                text,
                layer_stride=int(kwargs.get("layer_stride") or 4),
                include_sae=include_sae,
            )
            report = lat.to_thought_report()
            report.metadata["requested_model"] = model
            report.metadata["resolved_model"] = str(mid)
            report.metadata["load_in_4bit"] = bool(fourbit)
            if include_probe_train:
                report.metadata["probe_skipped"] = (
                    "Skipped synthetic probe train on HF residual path."
                )
            return report
        except Exception as exc:
            last_exc = exc
            continue
    assert last_exc is not None
    raise last_exc
