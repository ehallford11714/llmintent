"""Offline lexical space for region intent documents.

Word unigrams, bigrams, and alphabetic character trigrams — the same
contract as flybrain.embed.lexical_features, without scipy sparse or a
remote embedder. Fitted on the atlas documents so cosine is in a closed
space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_ALPHA = re.compile(r"^[a-z]+$")


def lexical_features(text: str) -> dict[str, float]:
    ws = _WORD.findall((text or "").lower())
    feats: dict[str, float] = {}

    def add(key: str, w: float) -> None:
        feats[key] = feats.get(key, 0.0) + w

    for w in ws:
        add(f"w:{w}", 1.0)
        if len(w) >= 4 and _ALPHA.match(w):
            padded = f"#{w}#"
            for i in range(len(padded) - 2):
                add(f"c:{padded[i:i + 3]}", 0.25)
    for a, b in zip(ws, ws[1:]):
        add(f"b:{a} {b}", 0.6)
    return feats


@dataclass
class LexicalSpace:
    vocab: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray | None = None
    fitted: bool = False

    def fit(self, documents: list[str]) -> "LexicalSpace":
        df: dict[str, int] = {}
        n = max(len(documents), 1)
        for doc in documents:
            for key in lexical_features(doc):
                df[key] = df.get(key, 0) + 1
        self.vocab = {k: i for i, k in enumerate(sorted(df))}
        idf = np.zeros(len(self.vocab), dtype=np.float64)
        for key, i in self.vocab.items():
            idf[i] = np.log((n + 1) / (df[key] + 1)) + 1.0
        self.idf = idf
        self.fitted = True
        return self

    def encode(self, text: str) -> np.ndarray:
        if not self.fitted or self.idf is None:
            raise RuntimeError("LexicalSpace.fit() first")
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        for key, w in lexical_features(text).items():
            idx = self.vocab.get(key)
            if idx is not None:
                vec[idx] += w * self.idf[idx]
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec /= n
        return vec

    def cosine(self, a: str, b: str) -> float:
        va, vb = self.encode(a), self.encode(b)
        return float(va @ vb)
