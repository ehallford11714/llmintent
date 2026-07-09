"""Text generation with optional retracement and focus steering."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmintent.heighten.intervention import extract_reasoning_focus_vector, steering_hooks
from llmintent.heighten.retrace import build_retrace_prompt
from llmintent.heighten.types import RetraceMode
from llmintent.live.session import LiveSession
from llmintent.models import ModelBundle
from llmintent.retracement.transformer import RetracementTransformer


def format_prompt(bundle: ModelBundle, prompt: str, *, use_chat: bool) -> str:
    """Apply chat template when the model supports it."""
    if not use_chat:
        return prompt
    tokenizer = bundle.tokenizer
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


@torch.no_grad()
def generate_completion(
    session: LiveSession,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    retracement_mode: str | None = None,
    steer: bool = False,
    anchor_prompt: str | None = None,
    use_chat: bool | None = None,
) -> tuple[str, bool]:
    """
    Greedy / sampled continuation with optional RetracementTransformer or steering.

    Returns (completion_text, steered_flag).
    """
    bundle = session.bundle
    use_chat = session.spec.chat_template if use_chat is None else use_chat
    text = format_prompt(bundle, prompt, use_chat=use_chat)

    mode = retracement_mode or session.config.retracement_mode
    steered = False

    retrace_prompt = build_retrace_prompt(
        anchor_prompt or prompt,
        mode=RetraceMode.EXPLICIT,
    )
    steer_vec = None
    steer_layers: list[int] = []
    if steer:
        steer_vec = extract_reasoning_focus_vector(bundle, prompt, retrace_prompt)
        steer_layers = [bundle.num_layers // 2, bundle.num_layers * 2 // 3]
        steered = True

    rt: RetracementTransformer | None = None
    if mode and mode != "baseline":
        session.set_retracement_mode(mode)
        rt = session.retracement

    enc = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
    input_ids = enc.input_ids

    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        ctx = generated

        def _forward(ids: torch.Tensor) -> torch.Tensor:
            if rt is not None:
                return rt.forward(ids)
            if steer_vec is not None:
                with steering_hooks(bundle, steer_layers, steer_vec, session.config.steering_coefficient):
                    out = bundle.model(ids)
                    return out.logits if hasattr(out, "logits") else out[0]
            out = bundle.model(ids)
            return out.logits if hasattr(out, "logits") else out[0]

        logits = _forward(ctx)
        next_logits = logits[0, -1, :]
        if temperature <= 0:
            next_id = int(next_logits.argmax())
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

        generated = torch.cat(
            [generated, torch.tensor([[next_id]], device=bundle.device)],
            dim=1,
        )
        if next_id == bundle.tokenizer.eos_token_id:
            break

    new_tokens = generated[0, input_ids.shape[1] :]
    completion = bundle.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return completion.strip(), steered


@torch.no_grad()
def next_token_topk(
    session: LiveSession,
    prompt: str,
    *,
    k: int = 5,
    retracement_mode: str | None = None,
) -> list[tuple[str, float]]:
    """Top-k next tokens — fastest real-time probe."""
    bundle = session.bundle
    text = format_prompt(bundle, prompt, use_chat=session.spec.chat_template)
    enc = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
    input_ids = enc.input_ids

    mode = retracement_mode or session.config.retracement_mode
    if mode and mode != "baseline":
        session.set_retracement_mode(mode)
        logits = session.retracement.forward(input_ids)
    else:
        out = bundle.model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]

    probs = F.softmax(logits[0, -1, :].float(), dim=-1)
    topk = torch.topk(probs, k)
    ids = topk.indices.tolist()
    return [
        (bundle.tokenizer.decode([i]), float(topk.values[j].item()))
        for j, i in enumerate(ids)
    ]
