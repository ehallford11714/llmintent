"""Latent thoughts from a real residual stream, decoded onto the fly atlas.

This is residual logit-lens + SAE occupancy + compile prior. It is not the
model's verbalized thinking channel, and it is not mind-reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from llmintent.anatomy.compile import compile_regions
from llmintent.anatomy.svd_map import match_text_to_region, svd_hidden_matrix
from llmintent.anatomy.atlas import region_ids
from llmintent.forward import forward_hidden_states, get_lm_head, normalize_hidden
from llmintent.latent_vendor.sae_lite import SAELite
from llmintent.latent_vendor.types import EPISTEMIC_CAVEATS, IntentHypothesis, LayerSaliency
from llmintent.models import ModelBundle


@dataclass
class LayerThought:
    layer: int
    depth: float
    band: str
    top_tokens: list[str]
    region: str
    region_score: float
    residual_l2: float

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "depth": round(self.depth, 4),
            "band": self.band,
            "top_tokens": list(self.top_tokens),
            "region": self.region,
            "region_score": round(self.region_score, 4),
            "residual_l2": round(self.residual_l2, 4),
        }


@dataclass
class LatentThoughtReport:
    text: str
    model_name: str
    compiled_regions: list[str]
    thoughts: list[LayerThought]
    occupancy: dict[str, float]
    hypothesized_intents: list[IntentHypothesis]
    layer_saliency: list[LayerSaliency]
    sae_features: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model_name": self.model_name,
            "compiled_regions": list(self.compiled_regions),
            "thoughts": [t.to_dict() for t in self.thoughts],
            "occupancy": {k: round(v, 4) for k, v in self.occupancy.items() if v > 0},
            "hypothesized_intents": [h.to_dict() for h in self.hypothesized_intents],
            "layer_saliency": [s.to_dict() for s in self.layer_saliency],
            "sae_features": self.sae_features,
            "notes": list(self.notes),
            "disclaimer": (
                "Layer thoughts are logit-lens tokens from residuals, mapped onto "
                "the fly-inspired atlas. Correlates, not decoded cognition."
            ),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Latent thoughts — `{self.model_name}`",
            "",
            f"**Prompt:** {self.text}",
            f"**Compile prior:** {', '.join(self.compiled_regions) or '(none)'}",
            "",
            "## Layer thoughts (logit lens → region)",
            "",
            "| Layer | Depth | Band | Region | Tokens |",
            "|------:|------:|------|--------|--------|",
        ]
        for t in self.thoughts:
            toks = ", ".join(t.top_tokens[:4]) or "—"
            lines.append(
                f"| {t.layer} | {t.depth:.2f} | {t.band} | `{t.region}` | {toks} |"
            )
        lines.append("")
        if self.occupancy:
            lines.append("## Residual occupancy")
            lines.append("")
            for rid, v in sorted(self.occupancy.items(), key=lambda kv: -kv[1]):
                if v <= 0:
                    continue
                lines.append(f"- `{rid}`: {v:.3f}")
            lines.append("")
        lines.append("## Caveat")
        lines.append(
            "Logit-lens tokens are next-token correlates at that layer, "
            "not a claim the model contains fly neuropils."
        )
        lines.append("")
        return "\n".join(lines)

    def to_thought_report(self):
        from llmintent.latent_vendor.report import ThoughtReport

        logit = [
            {
                "layer": t.layer,
                "top_tokens": [{"token": tok} for tok in t.top_tokens],
                "region": t.region,
                "region_score": round(t.region_score, 4),
                "band": t.band,
                "depth": round(t.depth, 4),
                "method": "logit_lens_residual",
            }
            for t in self.thoughts
        ]
        caveats = list(EPISTEMIC_CAVEATS) + [
            "Logit-lens tokens are next-token correlates at that residual, not inner speech.",
            "Region labels are cosine matches onto fly-inspired intent documents.",
        ]
        return ThoughtReport(
            text=self.text,
            backend="hf",
            hypothesized_intents=self.hypothesized_intents,
            layer_saliency=self.layer_saliency,
            logit_lens=logit,
            sae_features=self.sae_features,
            model_name=self.model_name,
            caveats=caveats,
            metadata={
                "mode": "residual_logit_lens",
                "compiled_regions": self.compiled_regions,
                "occupancy": {k: round(v, 4) for k, v in self.occupancy.items() if v > 0},
                "notes": self.notes,
                "source": "llmintent.anatomy.thoughts",
            },
        )


def _band_for_depth(depth: float) -> str:
    if depth < 0.34:
        return "sensory"
    if depth < 0.72:
        return "central"
    return "motor"


def _sample_layers(n: int, stride: int) -> list[int]:
    if n <= 0:
        return []
    idxs = list(range(0, n, max(int(stride), 1)))
    if (n - 1) not in idxs:
        idxs.append(n - 1)
    return idxs


def _decode_hidden(bundle: ModelBundle, hidden: torch.Tensor, top_k: int) -> list[str]:
    h = hidden.detach().float()
    if h.dim() == 1:
        h = h.unsqueeze(0)
    try:
        vec = h[0] if h.dim() > 1 else h
        try:
            vec = normalize_hidden(bundle, vec)
        except Exception:
            pass
        head = get_lm_head(bundle)
        params = list(head.parameters())
        dev = params[0].device if params else vec.device
        logits = head(vec.unsqueeze(0).to(dev))
        if logits.dim() == 2:
            logits = logits[0]
        elif logits.dim() > 2:
            logits = logits.reshape(-1, logits.shape[-1])[-1]
    except Exception:
        from llmintent.models import get_unembedding_matrix

        unembed = get_unembedding_matrix(bundle.model).float()
        vec = h[0] if h.dim() > 1 else h
        try:
            vec = normalize_hidden(bundle, vec)
        except Exception:
            pass
        logits = F.linear(vec.float().cpu(), unembed.cpu())
    k = min(top_k, int(logits.numel()))
    ids = torch.topk(logits.float(), k=k).indices.tolist()
    tokens: list[str] = []
    for tid in ids:
        piece = bundle.tokenizer.decode([int(tid)], skip_special_tokens=True).strip()
        if piece:
            tokens.append(piece)
    return tokens


def inspect_latent_thoughts(
    bundle: ModelBundle,
    text: str,
    *,
    layer_stride: int = 4,
    top_k: int = 6,
    include_sae: bool = True,
) -> LatentThoughtReport:
    """Forward the prompt, logit-lens sampled layers, map tokens onto atlas regions."""
    plan = compile_regions(text)
    _, states = forward_hidden_states(bundle, text)
    # states[0] = embeddings; block i = states[i+1]
    n_blocks = max(len(states) - 1, 1)
    sample = _sample_layers(n_blocks, layer_stride)

    thoughts: list[LayerThought] = []
    occupancy = {rid: 0.0 for rid in region_ids()}
    saliency: list[LayerSaliency] = []
    last_rows: list[np.ndarray] = []
    hyps: list[IntentHypothesis] = []

    for iso in plan.hits:
        hyps.append(
            IntentHypothesis(
                tag=iso.region,
                score=float(iso.score),
                method="compile_intent_doc",
                evidence=iso.span,
                confidence="medium",
            )
        )

    for li in sample:
        idx = min(li + 1, len(states) - 1)
        hidden = states[idx][0, -1, :]
        l2 = float(torch.linalg.vector_norm(hidden.float()).cpu())
        tokens = _decode_hidden(bundle, hidden, top_k)
        blob = " ".join(tokens)
        rid, score = match_text_to_region(blob) if blob else ("workspace", 0.0)
        # Weak cosine onto intent docs is not occupancy — same floor as compile.
        if score < 0.18:
            rid = "workspace"
        else:
            occupancy[rid] += max(score, 0.0)
        depth = li / max(n_blocks - 1, 1)
        thoughts.append(
            LayerThought(
                layer=li,
                depth=depth,
                band=_band_for_depth(depth),
                top_tokens=tokens,
                region=rid,
                region_score=float(score),
                residual_l2=l2,
            )
        )
        saliency.append(
            LayerSaliency(layer=li, score=l2, source="residual_l2_last_token")
        )
        hyps.append(
            IntentHypothesis(
                tag=rid,
                score=float(score),
                method="logit_lens_region",
                layer=li,
                evidence=" | ".join(tokens[:4]),
                confidence="low",
            )
        )
        last_rows.append(hidden.detach().float().cpu().numpy().reshape(-1))

    total = sum(occupancy.values()) or 1.0
    occupancy = {k: v / total for k, v in occupancy.items()}

    sae_features = None
    if include_sae and len(last_rows) >= 4:
        H = np.stack(last_rows, axis=0)
        n_comp = min(12, H.shape[0], max(H.shape[1] // 64, 4))
        try:
            sae = SAELite(n_components=n_comp, random_state=1, max_iter=20)
            sae_features = sae.fit(H).encode(H).to_dict()
            sae_features["note"] = "SAE-lite on last-token residuals of sampled layers."
        except Exception as exc:
            sae_features = {"error": str(exc)}
        try:
            _, S, Vh = svd_hidden_matrix(H, top_k=min(6, H.shape[0]))
            _ = (S, Vh)
        except Exception:
            pass

    notes = [
        f"Sampled {len(sample)} / {n_blocks} blocks (stride={layer_stride}).",
        "Logit lens is a next-token correlate at that residual, not inner speech.",
        "Qwen thinking tokens (if any) are the verbal channel; this is the latent one.",
    ]
    if plan.dropped:
        notes.append("Compile dropped: " + "; ".join(plan.dropped[:4]))

    # squash duplicate compile+lens hyps by keeping highest score per tag
    best: dict[str, IntentHypothesis] = {}
    for h in hyps:
        key = f"{h.tag}:{h.method}"
        if key not in best or h.score > best[key].score:
            best[key] = h

    return LatentThoughtReport(
        text=text,
        model_name=bundle.name,
        compiled_regions=plan.regions,
        thoughts=thoughts,
        occupancy=occupancy,
        hypothesized_intents=list(best.values()),
        layer_saliency=saliency,
        sae_features=sae_features,
        notes=notes,
    )


__all__ = [
    "LatentThoughtReport",
    "LayerThought",
    "inspect_latent_thoughts",
]
