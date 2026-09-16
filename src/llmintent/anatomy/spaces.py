"""Architecture-aware SVD: FFN down-projection convention and signed logits.

For an FFN, z(x) is post-nonlinearity (and post-gate) of width m.
W_down has shape [d, m] and maps into residual space:

    W_down = U S Vᵀ
    u_k : residual write direction (column of U)
    v_k : loadings over actual FFN units (column of V)
    a_k(x) = s_k v_kᵀ z(x)
    residual contribution = a_k(x) u_k

An SVD component is a combination of units, not one neuron.
Both signs of each direction are analyzed. Token projections are
candidate-generating signals, not functional labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llmintent.anatomy.evidence import ACCEPTANCE_FLOOR, UNIDENTIFIED, package_versions


def down_weight_d_by_m(weight: Any, layout: str) -> np.ndarray:
    """Return W_down with shape [d, m] (residual × FFN units)."""
    from llmintent.anatomy.weights import extract_dense_weight

    if isinstance(weight, np.ndarray):
        w = np.asarray(weight, dtype=np.float64)
    else:
        w = extract_dense_weight(weight)
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("weight must be 2D")
    if layout == "conv1d_in_out":
        # GPT-2 Conv1D: y = z @ W with W [m, d] → W_down = W.T [d, m]
        return w.T
    if layout == "linear_out_in":
        # nn.Linear: y = z @ W.T with W already [out, in] = [d, m]
        return w
    # Fallback: make first dim residual (smaller for typical FFN).
    return w if w.shape[0] <= w.shape[1] else w.T


def svd_down(w_down: np.ndarray, *, top_k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W_down [d, m] = U S Vᵀ. Returns U [d,k], S [k], V [m,k]."""
    W = np.asarray(w_down, dtype=np.float64)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    k = W.shape[0] if top_k is None else min(int(top_k), U.shape[1], len(S))
    return U[:, :k], S[:k], Vt[:k, :].T


def reconstruction_error(w_down: np.ndarray, U: np.ndarray, S: np.ndarray, V: np.ndarray) -> float:
    approx = (U * S) @ V.T
    num = float(np.linalg.norm(w_down - approx))
    den = float(np.linalg.norm(w_down)) + 1e-12
    return num / den


def coefficient(s_k: float, v_k: np.ndarray, z: np.ndarray) -> float:
    return float(s_k * np.dot(v_k, z))


def sign_invariant_subspace_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Principal angles overlap in [0, 1] for column-subspaces."""
    qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(B, dtype=np.float64))
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(s**2))


@dataclass
class SVDComponent:
    layer: int
    index: int
    singular_value: float
    u: np.ndarray
    v: np.ndarray
    unit_loadings: list[tuple[int, float]]
    reconstruction_error: float
    module_path: str
    layout: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self, *, include_vectors: bool = False) -> dict[str, Any]:
        payload = {
            "layer": self.layer,
            "index": self.index,
            "singular_value": round(float(self.singular_value), 6),
            "top_units": [{"unit": i, "loading": round(v, 4)} for i, v in self.unit_loadings[:16]],
            "reconstruction_error": round(self.reconstruction_error, 6),
            "module_path": self.module_path,
            "layout": self.layout,
            "notes": list(self.notes),
        }
        if include_vectors:
            payload["u"] = [round(float(x), 6) for x in self.u[:32]]
            payload["v_preview"] = [round(float(x), 6) for x in self.v[:32]]
        return payload


def decompose_ffn_down(
    weight: Any,
    *,
    layout: str,
    layer: int,
    module_path: str,
    top_k: int = 8,
    unit_topk: int = 8,
) -> list[SVDComponent]:
    W = down_weight_d_by_m(weight, layout)
    U, S, V = svd_down(W, top_k=top_k)
    err = reconstruction_error(W, U, S, V)
    out: list[SVDComponent] = []
    for k in range(len(S)):
        v = V[:, k]
        order = np.argsort(-np.abs(v))[:unit_topk]
        out.append(
            SVDComponent(
                layer=layer,
                index=k,
                singular_value=float(S[k]),
                u=U[:, k],
                v=v,
                unit_loadings=[(int(i), float(v[i])) for i in order],
                reconstruction_error=err,
                module_path=module_path,
                layout=layout,
                notes=[
                    "u is residual write; v loads FFN units; not one biological/artificial neuron.",
                    "Bias is not included in this decomposition.",
                ],
            )
        )
    return out


def signed_logit_profile(
    u: np.ndarray,
    unembed: Any,
    tokenizer: Any,
    *,
    top_k: int = 8,
    contrast_ids: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """W_vocab u and −u. Token strings are annotations, not labels."""
    import torch

    vec = torch.as_tensor(u, dtype=torch.float32)
    if hasattr(unembed, "detach"):
        W = unembed.detach().float().cpu()
    else:
        W = torch.as_tensor(unembed, dtype=torch.float32)
    if W.shape[1] != vec.numel():
        return {
            "source": "unavailable",
            "reason": f"unembed {tuple(W.shape)} incompatible with residual {int(vec.numel())}",
            "plus": [],
            "minus": [],
        }
    logits = W @ vec
    def _top(sign: int) -> list[dict[str, Any]]:
        vals, idx = torch.topk(sign * logits, k=min(top_k, logits.numel()))
        rows = []
        for score, tid in zip(vals.tolist(), idx.tolist()):
            tok = tokenizer.decode([int(tid)], skip_special_tokens=True) if tokenizer is not None else str(tid)
            rows.append({"token": tok.strip() or f"id:{tid}", "id": int(tid), "logit": round(float(sign * score), 4)})
        return rows

    contrasts: dict[str, dict[str, float]] = {}
    if contrast_ids:
        for name, ids in contrast_ids.items():
            plus = float(sum(logits[i].item() for i in ids if 0 <= i < logits.numel()))
            contrasts[name] = {"plus": plus, "minus": -plus}
    return {
        "source": "inferred",
        "method": "W_vocab @ u  (logit lens / candidate generator)",
        "plus": _top(1),
        "minus": _top(-1),
        "contrasts": contrasts,
        "note": "Not the exact final-output effect of intervening at this layer.",
    }


def contextual_lens(
    hidden: Any,
    direction: np.ndarray,
    unembed: Any,
    *,
    alpha: float,
    norm_fn,
) -> dict[str, Any]:
    """W_vocab [N(h+αu) − N(h)] at a documented site."""
    import torch

    h = hidden.detach().float() if hasattr(hidden, "detach") else torch.as_tensor(hidden, dtype=torch.float32)
    u = torch.as_tensor(direction, dtype=torch.float32, device=h.device)
    base = norm_fn(h)
    steered = norm_fn(h + float(alpha) * u.to(h.device))
    W = unembed.detach().float() if hasattr(unembed, "detach") else torch.as_tensor(unembed, dtype=torch.float32)
    W = W.to(base.device)
    delta = W @ (steered - base)
    return {
        "alpha": float(alpha),
        "delta_norm": float(torch.linalg.vector_norm(delta).cpu()),
        "source": "inferred",
        "note": "Logit-lens estimate at this site; remaining network not run.",
    }


def match_or_abstain(score: float, label: str, *, floor: float = ACCEPTANCE_FLOOR) -> tuple[str, float]:
    if score < floor:
        return UNIDENTIFIED, float(score)
    return label, float(score)


def random_control_axis(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=int(dim))
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).astype(np.float64)


def pin_versions() -> dict[str, str]:
    return package_versions()
