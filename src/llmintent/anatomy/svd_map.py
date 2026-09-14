"""SVD mapping from transformer subspaces onto atlas regions.

Two paths:

* **weights** — FFN SVD per layer, unembed top tokens, lexical-match the
  token string onto region intent documents. Structural anatomy.
* **activations** — SVD of the layer × hidden matrix for one prompt, then
  cosine of each right singular vector against region probe centroids
  (or against planted axes in the offline synthetic path).

Both return occupancy: region → how much of this model/prompt lives there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llmintent.anatomy.atlas import REGIONS, layers_for_region, region_ids
from llmintent.anatomy.compile import compile_regions
from llmintent.anatomy.lexical import LexicalSpace, lexical_features
from llmintent.svd import perform_svd_on_ffn


@dataclass
class ComponentMap:
    index: int
    region: str
    score: float
    layer: int | None = None
    top_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "region": self.region,
            "score": round(self.score, 4),
            "layer": self.layer,
            "top_tokens": list(self.top_tokens),
        }


@dataclass
class SVDAnatomy:
    """Region occupancy recovered from SVD components."""

    kind: str
    occupancy: dict[str, float]
    components: list[ComponentMap] = field(default_factory=list)
    layer_region: dict[int, str] = field(default_factory=dict)
    axes: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "occupancy": {k: round(v, 4) for k, v in self.occupancy.items() if v > 0},
            "components": [c.to_dict() for c in self.components[:24]],
            "layer_region": {str(k): v for k, v in self.layer_region.items()},
            "notes": list(self.notes),
        }


def _intent_space() -> tuple[LexicalSpace, dict[str, np.ndarray]]:
    docs = [r.intent + " " + r.probe for r in REGIONS]
    space = LexicalSpace().fit(docs)
    mat = {r.id: space.encode(r.intent + " " + r.probe) for r in REGIONS}
    return space, mat


def match_text_to_region(text: str) -> tuple[str, float]:
    space, mat = _intent_space()
    q = space.encode(text)
    best_id = "workspace"
    best = -1.0
    for rid, vec in mat.items():
        s = float(q @ vec)
        if s > best:
            best, best_id = s, rid
    return best_id, best


def svd_hidden_matrix(H: np.ndarray, *, top_k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H is [n_obs, hidden]. Returns U, S, Vh with Vh [k, hidden]."""
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError("H must be 2D")
    k = min(H.shape)
    if top_k is not None:
        k = min(k, int(top_k))
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    return U[:, :k], S[:k], Vh[:k]


def map_synthetic(
    H: np.ndarray,
    axes: dict[str, np.ndarray],
    *,
    layer_index: list[int] | None = None,
) -> SVDAnatomy:
    """Assign SVD components of H to the nearest planted region axis."""
    _, S, Vh = svd_hidden_matrix(H)
    names = list(axes)
    A = np.stack([_unit(axes[n]) for n in names], axis=0)
    components: list[ComponentMap] = []
    occupancy = {rid: 0.0 for rid in region_ids()}
    recovered: dict[str, np.ndarray] = {}
    for i, (s, v) in enumerate(zip(S, Vh)):
        v = _unit(v)
        scores = A @ v
        j = int(np.argmax(np.abs(scores)))
        rid = names[j]
        score = abs(float(scores[j])) * float(s)
        occupancy[rid] += score
        recovered[rid] = v
        components.append(
            ComponentMap(index=i, region=rid, score=float(abs(scores[j])), layer=None)
        )
    total = sum(occupancy.values()) or 1.0
    occupancy = {k: v / total for k, v in occupancy.items()}
    layer_region: dict[int, str] = {}
    if layer_index is not None and H.shape[0] == len(layer_index):
        for row, layer in zip(H, layer_index):
            u = _unit(row)
            scores = {n: abs(float(_unit(axes[n]) @ u)) for n in names}
            layer_region[int(layer)] = max(scores, key=scores.get)
    return SVDAnatomy(
        kind="synthetic",
        occupancy=occupancy,
        components=components,
        layer_region=layer_region,
        axes=recovered,
        notes=["SVD of planted hidden matrix; axes are synthetic, not model weights."],
    )


def map_activations(
    bundle: Any,
    prompt: str,
    *,
    top_k: int = 8,
) -> SVDAnatomy:
    """SVD of per-layer last-token hiddens, matched to region probe centroids."""
    import torch

    from llmintent.forward import forward_hidden_states

    centroids = _probe_centroids(bundle)
    _, states = forward_hidden_states(bundle, prompt)
    # skip embedding row 0
    rows = []
    for state in states[1:]:
        h = state[0, -1, :].detach().float().cpu().numpy()
        rows.append(h)
    H = np.stack(rows, axis=0)
    _, S, Vh = svd_hidden_matrix(H, top_k=min(top_k, H.shape[0]))
    names = list(centroids)
    A = np.stack([centroids[n] for n in names], axis=0)
    occupancy = {rid: 0.0 for rid in region_ids()}
    components: list[ComponentMap] = []
    recovered: dict[str, np.ndarray] = {}
    for i, (s, v) in enumerate(zip(S, Vh)):
        v = _unit(v)
        scores = A @ v
        j = int(np.argmax(np.abs(scores)))
        rid = names[j]
        occupancy[rid] += abs(float(scores[j])) * float(s)
        recovered[rid] = v.astype(np.float64)
        components.append(
            ComponentMap(index=i, region=rid, score=float(abs(scores[j])))
        )
    total = sum(occupancy.values()) or 1.0
    occupancy = {k: v / total for k, v in occupancy.items()}

    layer_region: dict[int, str] = {}
    for li, row in enumerate(H):
        u = _unit(row)
        scores = {n: abs(float(centroids[n] @ u)) for n in names}
        layer_region[li] = max(scores, key=scores.get)

    plan = compile_regions(prompt)
    notes = [
        "Activation SVD on last-token residual per layer. Probe centroids from region.probe.",
        f"Compile prior regions: {plan.regions or '(none)'}.",
    ]
    _ = torch  # keep import used for dtype consistency at call sites
    return SVDAnatomy(
        kind="activations",
        occupancy=occupancy,
        components=components,
        layer_region=layer_region,
        axes=recovered,
        notes=notes,
    )


def map_weights(bundle: Any, *, top_k: int = 6) -> SVDAnatomy:
    """FFN SVD → unembed top tokens → lexical region match."""
    import torch
    import torch.nn.functional as F

    from llmintent.models import get_ffn_weight, get_transformer_layers, get_unembedding_matrix

    layers = get_transformer_layers(bundle.model)
    unembed = get_unembedding_matrix(bundle.model).float()
    occupancy = {rid: 0.0 for rid in region_ids()}
    components: list[ComponentMap] = []
    layer_region: dict[int, str] = {}
    tok = bundle.tokenizer

    with torch.no_grad():
        for li, layer in enumerate(layers):
            try:
                w = get_ffn_weight(layer)
            except AttributeError:
                continue
            top_v = perform_svd_on_ffn(w, top_k=top_k)
            votes: dict[str, float] = {}
            for ci in range(top_v.shape[1]):
                vec = top_v[:, ci]
                logits = F.linear(vec.to(unembed.device), unembed)
                top = torch.topk(logits, k=min(8, logits.numel()))
                tokens = []
                for tid in top.indices.tolist():
                    piece = tok.decode([tid], skip_special_tokens=True).strip()
                    if piece:
                        tokens.append(piece)
                blob = " ".join(tokens)
                rid, score = match_text_to_region(blob)
                votes[rid] = votes.get(rid, 0.0) + score
                occupancy[rid] += score
                components.append(
                    ComponentMap(
                        index=ci,
                        region=rid,
                        score=score,
                        layer=li,
                        top_tokens=tokens[:5],
                    )
                )
            if votes:
                layer_region[li] = max(votes, key=votes.get)

    total = sum(occupancy.values()) or 1.0
    occupancy = {k: v / total for k, v in occupancy.items()}
    return SVDAnatomy(
        kind="weights",
        occupancy=occupancy,
        components=components,
        layer_region=layer_region,
        notes=[
            "FFN SVD components unembedded to tokens, then lexically matched to region intent docs.",
            "Structural map — independent of the prompt.",
        ],
    )


def _probe_centroids(bundle: Any) -> dict[str, np.ndarray]:
    from llmintent.forward import forward_hidden_states

    out: dict[str, np.ndarray] = {}
    n = max(int(bundle.num_layers), 1)
    for region in REGIONS:
        _, states = forward_hidden_states(bundle, region.probe)
        idxs = layers_for_region(region.id, n)
        vecs = []
        for li in idxs:
            # states[0] is embeddings; block i is states[i+1]
            idx = min(li + 1, len(states) - 1)
            h = states[idx][0, -1, :].detach().float().cpu().numpy()
            vecs.append(_unit(h))
        out[region.id] = _unit(np.mean(np.stack(vecs), axis=0))
    return out


def region_axis_from_anatomy(anatomy: SVDAnatomy, region_id: str, dim: int) -> np.ndarray:
    if region_id in anatomy.axes and anatomy.axes[region_id].shape[-1] == dim:
        return _unit(anatomy.axes[region_id])
    rng = np.random.default_rng(abs(hash(region_id)) % (2**32))
    return _unit(rng.normal(size=dim))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


# lexical_features imported for callers that want token-blob features
__all__ = [
    "ComponentMap",
    "SVDAnatomy",
    "lexical_features",
    "map_activations",
    "map_synthetic",
    "map_weights",
    "match_text_to_region",
    "region_axis_from_anatomy",
    "svd_hidden_matrix",
]
