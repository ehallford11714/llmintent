"""Rule / heuristic backend for offline intent tagging."""

from __future__ import annotations

import re
from typing import Any

from llmintent.latent_vendor.types import IntentHypothesis, LayerSaliency

_RULES: list[tuple[str, list[str], float]] = [
    ("request_info", [r"\b(what|who|where|when|why|how|explain|tell me|describe)\b", r"\?"], 1.0),
    ("request_action", [r"\b(please|could you|can you|would you|do this|make|create|write|fix)\b"], 0.9),
    ("express_goal", [r"\b(i want|i need|my goal|aim to|intend to|so that|in order to)\b"], 1.0),
    ("express_constraint", [r"\b(cannot|can't|must not|don't|without|unless|only if|constraint|limit)\b"], 0.95),
    ("refuse_or_hedge", [r"\b(maybe|perhaps|not sure|i refuse|won't|cannot help|as an ai)\b"], 0.85),
    ("plan_or_reason", [r"\b(first|then|next|because|therefore|step|plan|reason|let's think)\b"], 0.9),
    ("social_affect", [r"\b(thanks|sorry|love|hate|happy|sad|angry|please|appreciate)\b"], 0.8),
]


def rule_label_text(text: str) -> list[IntentHypothesis]:
    low = text.lower()
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for tag, patterns, weight in _RULES:
        hits: list[str] = []
        raw = 0.0
        for pat in patterns:
            for m in re.finditer(pat, low, flags=re.I):
                hits.append(m.group(0))
                raw += weight
        if hits:
            scores[tag] = min(1.0, raw / (2.0 + len(hits)))
            evidence[tag] = hits[:5]
    if not scores:
        scores["other"] = 0.35
        evidence["other"] = ["no_rule_match"]

    total = sum(scores.values()) or 1.0
    hyps: list[IntentHypothesis] = []
    for tag, sc in sorted(scores.items(), key=lambda x: -x[1]):
        norm = sc / total
        conf = "medium" if norm >= 0.25 and tag != "other" else "low"
        hyps.append(
            IntentHypothesis(
                tag=tag,
                score=float(norm),
                method="rule_heuristic",
                evidence=", ".join(evidence.get(tag, [])),
                confidence=conf,
            )
        )
    return hyps


def heuristic_layer_saliency(text: str, n_layers: int = 6) -> list[LayerSaliency]:
    low = text.lower()
    plan = len(re.findall(r"\b(because|therefore|first|then|step|plan)\b", low))
    q = 1 if "?" in text else 0
    base = [0.1] * n_layers
    base[0] += 0.2 + 0.05 * min(len(text) / 80.0, 1.0)
    mid = min(n_layers - 1, n_layers // 2)
    base[mid] += 0.15 * (1 + plan)
    base[-1] += 0.1 + 0.15 * q + 0.05 * plan
    s = sum(base) or 1.0
    return [
        LayerSaliency(
            layer=i,
            score=float(base[i] / s),
            source="rule_heuristic",
            details={"note": "Not causal; placeholder saliency for offline reports."},
        )
        for i in range(n_layers)
    ]


def describe_backend() -> dict[str, Any]:
    return {
        "name": "rule",
        "offline": True,
        "tags": [t for t, _, _ in _RULES] + ["other"],
        "note": "Regex heuristics only — not model internals.",
    }
