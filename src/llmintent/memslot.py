"""Extra memory token: the lock as a register the stack can attend to.

Painting residuals failed. Directing the mouth stuttered. This inserts one
token just before the last prompt piece. Its embedding is the locked
BoundState, scaled to the prompt's RMS. Later tokens attend to it. The last
token stays the chat prefix, not the memory.

This is an added slot, not an in-stream residual write. Decay is not trained.
"""

from __future__ import annotations

from typing import Any

import torch


def memory_token_id(tokenizer: Any) -> int:
    for piece in (";", ".", ","):
        ids = tokenizer.encode(piece, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    eos = getattr(tokenizer, "eos_token_id", None)
    return int(eos) if eos is not None else 0


def insert_memory_ids(input_ids: torch.Tensor, mem_tid: int) -> tuple[torch.Tensor, int]:
    """Put a placeholder before the last token. Returns (ids, memory index)."""
    ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
    if ids.shape[1] < 1:
        raise ValueError("input_ids is empty")
    left = ids[:, :-1]
    last = ids[:, -1:]
    mid = torch.full(
        (ids.size(0), 1),
        int(mem_tid),
        device=ids.device,
        dtype=ids.dtype,
    )
    out = torch.cat([left, mid, last], dim=1)
    return out, int(left.shape[1])


def attach_memory_slot(
    bundle: Any,
    input_ids: torch.Tensor,
    vector: torch.Tensor,
    *,
    gain: float = 1.0,
):
    """Insert the lock as an embedding at the memory index. Caller must remove the hook."""
    ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
    ids = ids.to(bundle.device)
    tid = memory_token_id(bundle.tokenizer)
    ids, pos = insert_memory_ids(ids, tid)
    emb = bundle.model.get_input_embeddings()
    vec = vector.detach().float().reshape(-1)
    if vec.numel() != int(emb.weight.shape[1]):
        raise ValueError(f"bound dim {vec.numel()} != embed {int(emb.weight.shape[1])}")

    def _hook(_module, _inp, out: torch.Tensor) -> torch.Tensor:
        if out.dim() != 3 or out.shape[1] <= pos:
            return out
        h = out.clone()
        rms = float(h.detach().float().norm(dim=-1).mean().item())
        u = vec.to(device=h.device, dtype=torch.float32)
        u = u / (u.norm() + 1e-8) * rms * float(gain)
        h[:, pos, :] = u.to(dtype=h.dtype)
        return h

    handle = emb.register_forward_hook(_hook)
    return ids, pos, handle
