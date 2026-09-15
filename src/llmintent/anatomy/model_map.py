"""Anatomy: per-layer responsibility and a complete graph for any weighted model.

Call ``Anatomy.from_pretrained(model_id)`` or ``Anatomy.from_bundle(bundle)``.
Weights (FFN SVD → tokens → full intent catalogue) say what each layer is
responsible for. The graph unions the fly-connectome prior with residual
stream edges and co-responsibility links. This is a map, not mind-reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmintent.anatomy.atlas import REGION_BY_ID, what_region_does
from llmintent.anatomy.connectome import literature_region_connectome
from llmintent.anatomy.intents import INTENT_BY_ID, intent_ids, score_blob
from llmintent.anatomy.svd_map import match_text_to_region

_BAND = {"sensory": 0, "central": 1, "motor": 2}


def _layer_band(i: int, n: int) -> str:
    """Optional depth prior/baseline — not a discovered functional organization."""
    if n <= 1:
        return "central"
    d = i / max(n - 1, 1)
    if d < 0.34:
        return "sensory"
    if d < 0.72:
        return "central"
    return "motor"


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    kind: str
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 4),
            "kind": self.kind,
            "evidence": self.evidence or self.kind,
        }


@dataclass
class LayerResponsibility:
    """What a residual block is responsible for (all intents)."""

    layer: int
    band: str
    intents: dict[str, float]
    tokens: list[str] = field(default_factory=list)
    source: str = "prior"

    @property
    def top(self) -> list[tuple[str, float]]:
        return sorted(
            ((k, v) for k, v in self.intents.items() if v > 0),
            key=lambda kv: -kv[1],
        )

    def responsible_for(self, k: int = 5) -> str:
        tops = self.top[:k]
        if not tops:
            return f"Layer {self.layer} ({self.band}) has no identified intent from weights."
        bits = ", ".join(f"`{i}` {s:.2f}" for i, s in tops)
        if self.source in {"prior", "weights"}:
            return (
                f"Layer {self.layer} ({self.band}) is associated with {bits} "
                "(token-projection candidate; unvalidated)."
            )
        return f"Layer {self.layer} ({self.band}) is associated with {bits}."

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "band": self.band,
            "source": self.source,
            "responsible_for": self.responsible_for(),
            "top": [{"intent": i, "score": round(s, 4)} for i, s in self.top[:12]],
            "intents": {k: round(v, 4) for k, v in self.intents.items()},
            "tokens": list(self.tokens[:12]),
        }


@dataclass
class AnatomyGraph:
    nodes: list[dict] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "nodes": list(self.nodes),
            "edges": [e.to_dict() for e in self.edges],
            "mermaid": self.mermaid(),
        }

    def mermaid(self) -> str:
        lines = ["flowchart LR"]
        shown: set[str] = set()
        # Prefer cascade + connectome + layer stream; cap for readability.
        ranked = sorted(self.edges, key=lambda e: -e.weight)
        kinds_keep = {"connectome_prior", "stream", "responsible", "cascade"}
        picked = [e for e in ranked if e.kind in kinds_keep][:80]
        for e in picked:
            for nid in (e.source, e.target):
                if nid in shown:
                    continue
                shown.add(nid)
                label = nid.replace('"', "'")
                lines.append(f'  {_nid(nid)}["{label}"]')
            lines.append(f"  {_nid(e.source)} -->|{e.kind}| {_nid(e.target)}")
        if len(lines) == 1:
            lines.append('  empty["no edges"]')
        return "\n".join(lines)


def _nid(name: str) -> str:
    return "n_" + "".join(ch if ch.isalnum() else "_" for ch in name)[:48]


def _empty_intents() -> dict[str, float]:
    return {iid: 0.0 for iid in intent_ids()}


def layer_responsibilities_from_weights(bundle: Any, *, top_k: int = 4) -> list[LayerResponsibility]:
    """FFN SVD of every block → tokens → full intent catalogue."""
    import torch
    import torch.nn.functional as F

    from llmintent.models import get_ffn_weight, get_transformer_layers, get_unembedding_matrix
    from llmintent.svd import perform_svd_on_ffn

    layers = get_transformer_layers(bundle.model)
    n = len(layers)
    unembed = get_unembedding_matrix(bundle.model).float()
    tok = bundle.tokenizer
    out: list[LayerResponsibility] = []
    with torch.no_grad():
        for li, layer in enumerate(layers):
            intents = _empty_intents()
            tokens: list[str] = []
            try:
                w = get_ffn_weight(layer)
            except AttributeError:
                out.append(
                    LayerResponsibility(li, _layer_band(li, n), intents, source="weights_missing")
                )
                continue
            top_v = perform_svd_on_ffn(w, top_k=top_k)
            for ci in range(top_v.shape[1]):
                vec = top_v[:, ci]
                logits = F.linear(vec.to(unembed.device), unembed)
                k = min(8, int(logits.numel()))
                ids = torch.topk(logits, k=k).indices.tolist()
                pieces: list[str] = []
                for tid in ids:
                    piece = tok.decode([int(tid)], skip_special_tokens=True).strip()
                    if piece:
                        pieces.append(piece)
                        if piece not in tokens:
                            tokens.append(piece)
                blob = " ".join(pieces)
                for iid, sc in score_blob(blob).items():
                    intents[iid] += sc
                rid, sc = match_text_to_region(blob)
                key = f"atlas.{rid}"
                if key in intents:
                    intents[key] += max(sc, 0.0)
            total = sum(intents.values()) or 1.0
            intents = {k: v / total for k, v in intents.items()}
            out.append(
                LayerResponsibility(
                    layer=li,
                    band=_layer_band(li, n),
                    intents=intents,
                    tokens=tokens[:16],
                    source="weights",
                )
            )
    return out


def layer_responsibilities_offline(text: str, *, n_layers: int = 12) -> list[LayerResponsibility]:
    """Prior-only map: paint compile/intent scores onto depth bands."""
    from llmintent.anatomy.intents import band_for_intent
    from llmintent.anatomy.trace import trace_prompt

    n = max(int(n_layers), 1)
    rows = [
        LayerResponsibility(i, _layer_band(i, n), _empty_intents(), source="prior")
        for i in range(n)
    ]
    overall = score_blob(text or "")
    for iid, val in overall.items():
        band = band_for_intent(iid)
        for row in rows:
            if row.band == band:
                row.intents[iid] = max(row.intents[iid], val)
    trace = trace_prompt(text or "")
    n_spans = max(len(trace.spans), 1)
    for span in trace.spans:
        lo = int(span.index / n_spans * n)
        hi = max(lo + 1, int((span.index + 1) / n_spans * n))
        sc = score_blob(span.text)
        for rid, occ in span.occupancy.items():
            if occ > 0:
                sc[f"atlas.{rid}"] = max(sc.get(f"atlas.{rid}", 0.0), occ)
        for i in range(lo, min(hi, n)):
            for iid, val in sc.items():
                if iid in rows[i].intents:
                    rows[i].intents[iid] = max(rows[i].intents[iid], val)
    for row in rows:
        tot = sum(row.intents.values())
        if tot > 0:
            row.intents = {k: v / tot for k, v in row.intents.items()}
    return rows


def complete_graph(layers: list[LayerResponsibility]) -> AnatomyGraph:
    """Union fly-connectome prior, residual stream, and learned responsibility."""
    nodes: list[dict] = []
    seen: set[str] = set()

    def add_node(nid: str, kind: str, **meta: Any) -> None:
        if nid in seen:
            return
        seen.add(nid)
        nodes.append({"id": nid, "kind": kind, **meta})

    for iid in intent_ids():
        spec = INTENT_BY_ID[iid]
        add_node(iid, "intent", family=spec.family, polarity=spec.polarity)
    for rid, region in REGION_BY_ID.items():
        add_node(f"atlas.{rid}", "atlas", band=region.band, does=what_region_does(rid))
    for row in layers:
        add_node(f"L{row.layer}", "layer", band=row.band, index=row.layer)

    edges: list[GraphEdge] = []
    conn = literature_region_connectome()
    for e in conn.edges:
        edges.append(
            GraphEdge(
                f"atlas.{e.source}",
                f"atlas.{e.target}",
                float(e.weight),
                "connectome_prior",
                evidence="literature_prior (not measured MaleCNS synapses)",
            )
        )
    for a, b in zip(layers, layers[1:]):
        edges.append(GraphEdge(f"L{a.layer}", f"L{b.layer}", 1.0, "stream", evidence="architecture residual order"))

    peak_layer: dict[str, int] = {}
    for row in layers:
        for iid, sc in row.top[:6]:
            edges.append(GraphEdge(f"L{row.layer}", iid, sc, "responsible", evidence="inferred token-projection association"))
            prev = peak_layer.get(iid)
            if prev is None or sc > (layers[prev].intents.get(iid, 0) if prev < len(layers) else 0):
                peak_layer[iid] = row.layer
        tops = [i for i, _ in row.top[:4]]
        for i, src in enumerate(tops):
            for tgt in tops[i + 1 :]:
                edges.append(GraphEdge(src, tgt, 0.4, "co_responsible", evidence="inferred co-occurrence of top associations"))

    ordered = sorted(peak_layer.items(), key=lambda kv: kv[1])
    for (a, la), (b, lb) in zip(ordered, ordered[1:]):
        if lb > la:
            edges.append(GraphEdge(a, b, 0.6, "cascade", evidence="inferred depth order of peak associations; not a causal cascade"))

    return AnatomyGraph(nodes=nodes, edges=edges)


@dataclass
class Anatomy:
    """Per-layer intent responsibility + complete graph for a (possibly weighted) model."""

    model_name: str | None
    n_layers: int
    layers: list[LayerResponsibility]
    graph: AnatomyGraph
    text: str = ""
    source: str = "prior"
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        text: str = "",
        load_in_4bit: bool = False,
        device: str | None = None,
        top_k: int = 4,
    ) -> "Anatomy":
        from llmintent.models import load_model_bundle
        from llmintent.suite import resolve_model_spec

        spec = resolve_model_spec(model=model, use_env=False)
        hf_id = spec.hf_id if spec is not None else model
        four = load_in_4bit or (spec is not None and getattr(spec, "size", None) == "27b")
        bundle = load_model_bundle(hf_id, load_in_4bit=four, device=device)
        return cls.from_bundle(bundle, text=text, top_k=top_k)

    @classmethod
    def from_bundle(
        cls,
        bundle: Any,
        *,
        text: str = "",
        top_k: int = 4,
    ) -> "Anatomy":
        layers = layer_responsibilities_from_weights(bundle, top_k=top_k)
        graph = complete_graph(layers)
        name = getattr(bundle, "name", None) or "model"
        notes = [
            f"Anatomy from FFN weights of `{name}` ({len(layers)} layers).",
            "Each layer's association is SVD-unembed tokens scored on the intent catalogue (unvalidated).",
            "Graph = literature-prior connectome ∪ residual stream ∪ co-responsibility ∪ cascade.",
            "Depth bands are an optional prior, not a discovered organization.",
            "Not a claim the weights contain fly neuropils.",
        ]
        if text:
            notes.append(f"Prompt provided: {text[:120]}")
        return cls(
            model_name=name,
            n_layers=len(layers),
            layers=layers,
            graph=graph,
            text=text or "",
            source="weights",
            notes=notes,
        )

    @classmethod
    def offline(cls, text: str = "", *, n_layers: int = 12) -> "Anatomy":
        layers = layer_responsibilities_offline(text, n_layers=n_layers)
        graph = complete_graph(layers)
        return cls(
            model_name="offline",
            n_layers=len(layers),
            layers=layers,
            graph=graph,
            text=text or "",
            source="prior",
            notes=[
                "Offline Anatomy paints compile/intent scores onto depth-band priors (not discovered organization).",
                "Pass a model with weights (Anatomy.from_pretrained) for FFN SVD associations.",
            ],
        )

    def layer(self, index: int) -> LayerResponsibility:
        return self.layers[int(index)]

    def to_dict(self) -> dict:
        return {
            "name": "Anatomy",
            "model": self.model_name,
            "n_layers": self.n_layers,
            "source": self.source,
            "text": self.text,
            "layers": [row.to_dict() for row in self.layers],
            "graph": self.graph.to_dict(),
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Anatomy",
            "",
            f"**Model:** `{self.model_name or 'offline'}`  ",
            f"**Layers:** {self.n_layers} · **source:** `{self.source}`  ",
            f"**Prompt:** {self.text or '(structural weights only)'}  ",
            "",
            "## How each layer is responsible",
            "",
            "| Layer | Band | Responsible for |",
            "|------:|------|-----------------|",
        ]
        for row in self.layers:
            tops = row.top[:5]
            cell = ", ".join(f"`{i}` {s:.2f}" for i, s in tops) or "—"
            lines.append(f"| {row.layer} | {row.band} | {cell} |")
        lines.append("")
        lines.append("## Complete graph")
        lines.append("")
        lines.append(f"{len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges.")
        lines.append("")
        lines.append("```mermaid")
        lines.append(self.graph.mermaid())
        lines.append("```")
        lines.append("")
        lines.append("## Caveats")
        for n in self.notes:
            lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)


__all__ = [
    "Anatomy",
    "AnatomyGraph",
    "GraphEdge",
    "LayerResponsibility",
    "complete_graph",
    "layer_responsibilities_from_weights",
    "layer_responsibilities_offline",
]
