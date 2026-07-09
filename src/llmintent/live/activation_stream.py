"""Real-time per-layer activation snapshots during token generation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

from llmintent.forward import forward_hidden_states_from_ids, get_lm_head, normalize_hidden
from llmintent.live.generate import format_prompt
from llmintent.live.session import LiveSession
from llmintent.metrics import cosine_intensity, shannon_entropy
from llmintent.poles import build_numerical_pole


@dataclass
class LayerActivationSnapshot:
    """Per-token layer activation profile from the last forward position."""

    step: int
    token: str
    token_id: int
    layer_metrics: pd.DataFrame
    cumulative_text: str

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "token": self.token,
            "token_id": self.token_id,
            "cumulative_text": self.cumulative_text,
            "layers": self.layer_metrics.to_dict(orient="records"),
        }


def layer_metrics_from_states(
    bundle,
    states: list[torch.Tensor],
    *,
    position: int = -1,
) -> pd.DataFrame:
    """Build lightweight per-layer metrics for the given hidden-state stack."""
    head = get_lm_head(bundle)
    pole = build_numerical_pole(bundle).to(bundle.device)
    rows: list[dict] = []
    for i, state in enumerate(states):
        hidden = normalize_hidden(bundle, state[0, position, :].float())
        logits = head(hidden)
        probs = F.softmax(logits, dim=-1)
        rows.append(
            {
                "layer": i,
                "entropy": shannon_entropy(probs),
                "intensity": cosine_intensity(hidden, pole),
            }
        )
    return pd.DataFrame(rows)


def snapshot_layer_activations(
    session: LiveSession,
    input_ids: torch.Tensor,
    *,
    position: int = -1,
) -> pd.DataFrame:
    """Forward pass on token ids and return per-layer activation metrics."""
    bundle = session.bundle
    states = forward_hidden_states_from_ids(bundle, input_ids)
    return layer_metrics_from_states(bundle, states, position=position)


@torch.no_grad()
def iter_generate_with_layer_activations(
    session: LiveSession,
    prompt: str,
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.7,
    retracement_mode: str | None = None,
    use_chat: bool | None = None,
) -> Iterator[LayerActivationSnapshot]:
    """
    Yield a layer activation snapshot after each generated token.

    Uses the backbone forward pass with ``output_hidden_states=True`` so layer
    activations reflect the live context at each decoding step.
    """
    bundle = session.bundle
    use_chat = session.spec.chat_template if use_chat is None else use_chat
    text = format_prompt(bundle, prompt, use_chat=use_chat)

    mode = retracement_mode or session.config.retracement_mode
    rt = None
    if mode and mode != "baseline":
        session.set_retracement_mode(mode)
        rt = session.retracement

    enc = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
    input_ids = enc.input_ids
    generated = input_ids.clone()
    prompt_len = input_ids.shape[1]

    # Initial snapshot on the prompt (step 0)
    init_metrics = snapshot_layer_activations(session, generated)
    yield LayerActivationSnapshot(
        step=0,
        token="",
        token_id=-1,
        layer_metrics=init_metrics,
        cumulative_text=bundle.tokenizer.decode(generated[0], skip_special_tokens=True),
    )

    for step in range(1, max_new_tokens + 1):
        if rt is not None:
            states = forward_hidden_states_from_ids(bundle, generated)
            logits = rt.forward(generated)
        else:
            out = bundle.model(generated, output_hidden_states=True)
            logits = out.logits if hasattr(out, "logits") else out[0]
            states = list(out.hidden_states)
        metrics = layer_metrics_from_states(bundle, states)

        next_logits = logits[0, -1, :]
        if temperature <= 0:
            next_id = int(next_logits.argmax())
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

        token_str = bundle.tokenizer.decode([next_id])
        generated = torch.cat(
            [generated, torch.tensor([[next_id]], device=bundle.device)],
            dim=1,
        )
        completion_ids = generated[0, prompt_len:]
        cumulative = bundle.tokenizer.decode(completion_ids, skip_special_tokens=True)

        yield LayerActivationSnapshot(
            step=step,
            token=token_str,
            token_id=next_id,
            layer_metrics=metrics,
            cumulative_text=cumulative,
        )

        if next_id == bundle.tokenizer.eos_token_id:
            break
