"""SAE-lite sparse dictionary stub (sklearn)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import DictionaryLearning, SparseCoder


@dataclass
class SAELiteResult:
    codes: np.ndarray
    n_features: int
    sparsity: float
    reconstruction_mse: float | None = None
    top_active: list[tuple[int, float]] = field(default_factory=list)
    method: str = "sae_lite_sklearn"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "sparsity": self.sparsity,
            "reconstruction_mse": self.reconstruction_mse,
            "top_active": [{"feature": i, "activation": float(a)} for i, a in self.top_active],
            "method": self.method,
            "metadata": self.metadata,
            "caveat": (
                "SAE-lite codes are exploratory sparse features, not validated "
                "monosemantic concepts."
            ),
        }


@dataclass
class SAELite:
    n_components: int = 16
    alpha: float = 1.0
    max_iter: int = 40
    random_state: int = 0
    _dict: np.ndarray | None = field(default=None, repr=False)
    _trained: bool = field(default=False, repr=False)

    def fit(self, X: np.ndarray) -> "SAELite":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_comp = min(self.n_components, X.shape[0], X.shape[1])
        dl = DictionaryLearning(
            n_components=n_comp,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            transform_algorithm="lasso_lars",
            transform_alpha=self.alpha,
        )
        dl.fit(X)
        self._dict = dl.components_
        self._trained = True
        return self

    def encode(self, x: np.ndarray, top_k: int = 5) -> SAELiteResult:
        x = np.asarray(x, dtype=np.float64)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        if not self._trained or self._dict is None:
            rng = np.random.default_rng(self.random_state)
            neighbors = x2 + rng.normal(scale=0.05, size=(max(8, self.n_components), x2.shape[1]))
            self.fit(np.vstack([x2, neighbors]))
        assert self._dict is not None
        coder = SparseCoder(
            dictionary=self._dict,
            transform_algorithm="lasso_lars",
            transform_alpha=self.alpha,
        )
        codes = coder.transform(x2)[0]
        recon = codes @ self._dict
        mse = float(np.mean((x2.reshape(-1) - recon.reshape(-1)) ** 2))
        nnz = float(np.mean(np.abs(codes) > 1e-6))
        idx = np.argsort(np.abs(codes))[::-1][:top_k]
        top_active = [(int(i), float(codes[int(i)])) for i in idx if abs(codes[int(i)]) > 1e-8]
        return SAELiteResult(
            codes=codes,
            n_features=int(codes.shape[0]),
            sparsity=nnz,
            reconstruction_mse=mse,
            top_active=top_active,
            metadata={"dictionary_shape": list(self._dict.shape)},
        )

    def fit_encode_synthetic(self, dim: int = 32, n_samples: int = 64, top_k: int = 5) -> SAELiteResult:
        rng = np.random.default_rng(self.random_state)
        X = rng.normal(size=(n_samples, dim))
        self.fit(X)
        return self.encode(X[0], top_k=top_k)
