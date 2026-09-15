"""Entity slots: persist span history as named objects, not last-token state.

A transformer last residual is next-word state. Entity persistence writes a
slot at birth (digit or content noun) and does not update it when the last
token moves. The slot *is* the entity for this overlay. Injecting slot
vectors into the last residual is how a persisted entity can reach the
dictionary at the mouth.

This does not claim the base model grew fly neuropils or a number organ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_DIGIT = re.compile(r"^\d+$")
_NOUNS = {
    "fox",
    "dog",
    "cat",
    "bird",
    "spider",
    "horse",
    "cow",
    "sheep",
    "apple",
    "egg",
    "box",
    "river",
    "girl",
    "boy",
    "car",
    "house",
    "tree",
    "book",
    "fish",
    "bear",
    "wolf",
}


@dataclass
class EntitySlot:
    """A locked residual written at an entity's birth position."""

    id: str
    kind: str
    token: str
    position: int
    vector: np.ndarray
    write_layer: int
    locked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "token": self.token,
            "position": self.position,
            "write_layer": self.write_layer,
            "locked": self.locked,
            "dim": int(self.vector.size),
        }


@dataclass
class PersistenceReport:
    text: str
    last_token: str
    slots: list[EntitySlot]
    last_vs_entity: dict[str, float]
    slot_vs_entity: dict[str, float]
    last_vs_slot: dict[str, float]
    persists: dict[str, bool]
    mouth_baseline: str
    mouth_with_slots: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "last_token": self.last_token,
            "slots": [s.to_dict() for s in self.slots],
            "last_vs_entity": {k: round(v, 4) for k, v in self.last_vs_entity.items()},
            "slot_vs_entity": {k: round(v, 4) for k, v in self.slot_vs_entity.items()},
            "last_vs_slot": {k: round(v, 4) for k, v in self.last_vs_slot.items()},
            "persists": dict(self.persists),
            "mouth_baseline": self.mouth_baseline,
            "mouth_with_slots": self.mouth_with_slots,
            "notes": list(self.notes),
        }


def _piece(tokenizer: Any, tid: int) -> str:
    return tokenizer.decode([int(tid)], skip_special_tokens=True)


def detect_entity_spans(tokenizer: Any, input_ids: torch.Tensor) -> list[dict[str, Any]]:
    """Birth positions: digit tokens and a small noun list. Consecutive digits merge."""
    ids = input_ids.view(-1).tolist()
    raw: list[dict[str, Any]] = []
    for i, tid in enumerate(ids):
        text = _piece(tokenizer, tid).strip()
        if not text:
            continue
        if _DIGIT.match(text):
            raw.append({"position": i, "token": text, "kind": "digit", "id": f"digit:{text}@{i}"})
        elif text.lower() in _NOUNS:
            raw.append(
                {
                    "position": i,
                    "token": text.lower(),
                    "kind": "noun",
                    "id": f"noun:{text.lower()}@{i}",
                }
            )
    merged: list[dict[str, Any]] = []
    for span in raw:
        if (
            merged
            and span["kind"] == "digit"
            and merged[-1]["kind"] == "digit"
            and span["position"] == merged[-1]["position"] + 1
        ):
            prev = merged[-1]
            token = prev["token"] + span["token"]
            merged[-1] = {
                "position": span["position"],
                "token": token,
                "kind": "digit",
                "id": f"digit:{token}@{span['position']}",
                "start": prev.get("start", prev["position"]),
            }
        else:
            span = dict(span)
            span["start"] = span["position"]
            merged.append(span)
    return merged


def write_slots(
    hidden_states: list[torch.Tensor],
    spans: list[dict[str, Any]],
    *,
    layer: int = -1,
) -> list[EntitySlot]:
    """Copy residual at birth; later positions do not overwrite (locked)."""
    idx = layer if layer >= 0 else len(hidden_states) - 1
    H = hidden_states[idx]
    if H.dim() == 3:
        H = H[0]
    slots: list[EntitySlot] = []
    for span in spans:
        pos = int(span["position"])
        vec = H[pos].detach().float().cpu().numpy().reshape(-1)
        slots.append(
            EntitySlot(
                id=str(span["id"]),
                kind=str(span["kind"]),
                token=str(span["token"]),
                position=pos,
                vector=vec,
                write_layer=idx,
                locked=True,
            )
        )
    return slots


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _token_embed(bundle: Any, piece: str) -> np.ndarray | None:
    from llmintent.models import get_input_embeddings

    ids = bundle.tokenizer.encode(piece, add_special_tokens=False)
    if not ids:
        ids = bundle.tokenizer.encode(f" {piece}", add_special_tokens=False)
    if not ids:
        return None
    E = get_input_embeddings(bundle.model)
    return E[int(ids[0])].detach().float().cpu().numpy().reshape(-1)


def _last_logits(bundle: Any, hidden: torch.Tensor) -> torch.Tensor:
    from llmintent.forward import get_lm_head, normalize_hidden

    h = normalize_hidden(bundle, hidden.float())
    head = get_lm_head(bundle)
    try:
        return head(h.to(dtype=next(head.parameters()).dtype)).float()
    except Exception:
        from llmintent.models import get_unembedding_matrix

        unembed = get_unembedding_matrix(bundle.model).float()
        return F.linear(h.cpu(), unembed.cpu()).float()


def _top_piece(bundle: Any, logits: torch.Tensor) -> str:
    tid = int(torch.argmax(logits).item())
    return bundle.tokenizer.decode([tid], skip_special_tokens=True).strip() or f"id:{tid}"


def persist_entities(
    bundle: Any,
    text: str,
    *,
    layer: int = -1,
    gain: float = 1.5,
    inject: bool = True,
) -> PersistenceReport:
    """Write locked slots from a raw (non-chat) forward. Contrast with last token."""
    from llmintent.forward import forward_hidden_states_from_ids, normalize_hidden

    ids = bundle.tokenizer(text, return_tensors="pt").input_ids.to(bundle.device)
    last_str = _piece(bundle.tokenizer, int(ids[0, -1].item()))
    states = forward_hidden_states_from_ids(bundle, ids)
    spans = detect_entity_spans(bundle.tokenizer, ids)
    slots = write_slots(states, spans, layer=layer)
    last = states[-1][0, -1, :].detach().float().cpu().numpy().reshape(-1)

    last_vs: dict[str, float] = {}
    slot_vs: dict[str, float] = {}
    last_vs_slot: dict[str, float] = {}
    persists: dict[str, bool] = {}
    for slot in slots:
        last_vs_slot[slot.id] = _cos(last, slot.vector)
        emb = _token_embed(bundle, slot.token)
        if emb is not None:
            last_vs[slot.id] = _cos(last, emb)
            slot_vs[slot.id] = _cos(slot.vector, emb)
        # Entity persists iff the locked slot still holds birth state
        # and the last token is not that state.
        persists[slot.id] = bool(slot.locked) and last_vs_slot[slot.id] < 0.5

    last_h = normalize_hidden(bundle, states[-1][0, -1, :].float())
    base_logits = _last_logits(bundle, last_h)
    mouth_base = _top_piece(bundle, base_logits)
    mouth_slot = mouth_base
    notes = [
        "Slots are locked copies of birth residuals. Persistence here is the overlay, "
        "not a claim the base model stored entities.",
        "Last-token residual remains next-word state.",
    ]
    if inject and slots:
        mix = np.mean(np.stack([s.vector for s in slots], axis=0), axis=0)
        v = torch.from_numpy(mix.astype(np.float32))
        v = v / (v.norm() + 1e-8)
        steered = last_h.cpu() + float(gain) * v
        mouth_slot = _top_piece(bundle, _last_logits(bundle, steered))
        notes.append(
            f"Injected mean slot (gain={gain}) into last residual before the mouth."
        )
    n_persist = sum(1 for v in persists.values() if v)
    notes.append(
        f"{n_persist}/{len(persists)} slots stay disjoint from the last token "
        "(cos last-vs-slot < 0.5): entity in the slot, not in now."
    )
    return PersistenceReport(
        text=text,
        last_token=last_str,
        slots=slots,
        last_vs_entity=last_vs,
        slot_vs_entity=slot_vs,
        last_vs_slot=last_vs_slot,
        persists=persists,
        mouth_baseline=mouth_base,
        mouth_with_slots=mouth_slot,
        notes=notes,
    )
