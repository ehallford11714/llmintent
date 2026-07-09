"""Minimal linear probe for synthetic intent tags (offline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from llmintent.latent_vendor.types import INTENT_TAGS, IntentHypothesis


@dataclass
class ProbeResult:
    tag_scores: dict[str, float]
    accuracy: float | None = None
    auroc: float | None = None
    layer: int | None = None
    method: str = "linear_probe"
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_hypotheses(self, k: int = 3) -> list[IntentHypothesis]:
        ranked = sorted(self.tag_scores.items(), key=lambda x: -x[1])[:k]
        out: list[IntentHypothesis] = []
        for tag, score in ranked:
            conf = "high" if score >= 0.7 else ("medium" if score >= 0.4 else "low")
            out.append(
                IntentHypothesis(
                    tag=tag,
                    score=float(score),
                    method=self.method,
                    layer=self.layer,
                    evidence="logistic regression over activation vector",
                    confidence=conf,
                )
            )
        return out


@dataclass
class LinearIntentProbe:
    tags: tuple[str, ...] = INTENT_TAGS
    C: float = 1.0
    max_iter: int = 500
    random_state: int = 0
    _clf: LogisticRegression | None = field(default=None, repr=False)
    _le: LabelEncoder | None = field(default=None, repr=False)
    _trained: bool = field(default=False, repr=False)

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "LinearIntentProbe":
        X = np.asarray(X, dtype=np.float64)
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(list(y))
        self._clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._clf.fit(X, y_enc)
        self._trained = True
        return self

    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        if not self._trained or self._clf is None or self._le is None:
            raise RuntimeError("Probe not trained")
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        probs = self._clf.predict_proba(x)[0]
        return {str(self._le.classes_[i]): float(probs[i]) for i in range(len(probs))}

    def evaluate(self, X: np.ndarray, y: Sequence[str]) -> dict[str, float]:
        if not self._trained or self._clf is None or self._le is None:
            raise RuntimeError("Probe not trained")
        X = np.asarray(X, dtype=np.float64)
        y_enc = self._le.transform(list(y))
        pred = self._clf.predict(X)
        out: dict[str, float] = {"accuracy": float(accuracy_score(y_enc, pred))}
        try:
            proba = self._clf.predict_proba(X)
            if proba.shape[1] == 2:
                out["auroc"] = float(roc_auc_score(y_enc, proba[:, 1]))
            else:
                out["auroc"] = float(
                    roc_auc_score(y_enc, proba, multi_class="ovr", average="macro")
                )
        except ValueError:
            out["auroc"] = float("nan")
        return out

    def fit_synthetic(
        self,
        n_samples: int = 200,
        dim: int = 32,
        test_size: float = 0.25,
    ) -> tuple["LinearIntentProbe", ProbeResult]:
        rng = np.random.default_rng(self.random_state)
        tags = list(self.tags)
        directions = rng.normal(size=(len(tags), dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
        X_list: list[np.ndarray] = []
        y_list: list[str] = []
        for _ in range(n_samples):
            ti = int(rng.integers(0, len(tags)))
            x = directions[ti] * (1.5 + rng.random()) + rng.normal(scale=0.35, size=dim)
            X_list.append(x)
            y_list.append(tags[ti])
        X = np.stack(X_list)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_list, test_size=test_size, random_state=self.random_state, stratify=y_list
        )
        self.fit(X_tr, y_tr)
        metrics = self.evaluate(X_te, y_te)
        demo = directions[0] * 1.8 + rng.normal(scale=0.1, size=dim)
        result = ProbeResult(
            tag_scores=self.predict_proba(demo),
            accuracy=metrics.get("accuracy"),
            auroc=metrics.get("auroc"),
            layer=0,
            metadata={"synthetic": True, "dim": dim},
        )
        return self, result


def score_vector_with_probe(probe: LinearIntentProbe, vector: np.ndarray) -> ProbeResult:
    return ProbeResult(tag_scores=probe.predict_proba(vector), method="linear_probe")
