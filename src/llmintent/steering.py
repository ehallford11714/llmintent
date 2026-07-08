"""Runtime steering and CoT intensity analysis."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F

from llmintent.metrics import cosine_intensity, kl_divergence, shannon_entropy
from llmintent.models import ModelBundle


def _forward_hidden_states(bundle: ModelBundle, text: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    inputs = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        if bundle.is_causal:
            if hasattr(bundle.model, "transformer"):
                outputs = bundle.model.transformer(
                    inputs.input_ids,
                    output_hidden_states=True,
                )
            else:
                outputs = bundle.model(
                    **inputs,
                    output_hidden_states=True,
                )
            states = list(outputs.hidden_states)
        else:
            outputs = bundle.model(**inputs, output_hidden_states=True)
            states = list(outputs.hidden_states)
    return inputs, states


def _lm_head(bundle: ModelBundle) -> torch.nn.Module:
    if hasattr(bundle.model, "lm_head"):
        return bundle.model.lm_head
    if hasattr(bundle.model, "cls"):
        return bundle.model.cls
    if hasattr(bundle.model, "vocab_projector"):
        return bundle.model.vocab_projector
    raise AttributeError("Model has no recognized language modeling head")


def analyze_steering_intensity(
    bundle: ModelBundle,
    prompt: str,
    pole: torch.Tensor,
) -> pd.DataFrame:
    """Layer-wise pole intensity for the final token (notebook: analyze_steering)."""
    _, states = _forward_hidden_states(bundle, prompt)
    rows: list[dict[str, float | int]] = []
    for i, state in enumerate(states):
        vec = state[0, -1, :]
        rows.append({"layer": i, "intensity": cosine_intensity(vec, pole.to(bundle.device))})
    return pd.DataFrame(rows)


def run_intensity_sweep(
    bundle: ModelBundle,
    prompts: dict[str, str],
    pole: torch.Tensor,
) -> pd.DataFrame:
    """Full layer sweep for multiple prompt labels (notebook: run_full_intensity_sweep)."""
    rows: list[dict[str, float | int | str]] = []
    for label, text in prompts.items():
        _, states = _forward_hidden_states(bundle, text)
        for i, state in enumerate(states):
            vec = state[0, -1, :]
            rows.append(
                {
                    "layer": i,
                    "prompt_type": label,
                    "intensity": cosine_intensity(vec, pole.to(bundle.device)),
                }
            )
    return pd.DataFrame(rows)


def compare_cot_intensity(
    bundle: ModelBundle,
    direct_prompt: str,
    cot_prompt: str,
    pole: torch.Tensor,
    pivot_layer: int = 18,
) -> dict[str, float]:
    """Compare direct vs CoT intensity at a pivot layer."""
    results: dict[str, float] = {}
    for label, text in {"Direct": direct_prompt, "CoT": cot_prompt}.items():
        _, states = _forward_hidden_states(bundle, text)
        idx = min(pivot_layer, len(states) - 1)
        vec = states[idx][0, -1, :]
        results[label] = cosine_intensity(vec, pole.to(bundle.device))
    return results


def calculate_pivot_entropy(
    bundle: ModelBundle,
    direct_text: str,
    cot_text: str,
    pivot_layer: int = 18,
) -> dict[str, float]:
    """Entropy at pivot layer for direct vs CoT (notebook: calculate_pivot_entropy)."""
    head = _lm_head(bundle)
    out: dict[str, float] = {}
    for label, text in {"Direct": direct_text, "CoT": cot_text}.items():
        _, states = _forward_hidden_states(bundle, text)
        idx = min(pivot_layer, len(states) - 1)
        hidden = states[idx][0, -1, :]
        logits = head(hidden)
        probs = F.softmax(logits, dim=-1)
        out[label] = shannon_entropy(probs)
    return out


def run_stress_test(
    bundle: ModelBundle,
    prompt_simple: str,
    prompt_complex: str,
) -> pd.DataFrame:
    """KL divergence between simple and complex prompts per layer."""
    _, states1 = _forward_hidden_states(bundle, prompt_simple)
    _, states2 = _forward_hidden_states(bundle, prompt_complex)
    head = _lm_head(bundle)
    rows: list[dict[str, float | int]] = []
    for i, (s1, s2) in enumerate(zip(states1, states2)):
        probs1 = F.softmax(head(s1[0, -1, :]), dim=-1)
        probs2 = F.softmax(head(s2[0, -1, :]), dim=-1)
        rows.append({"layer": i, "kl_divergence": kl_divergence(probs2, probs1)})
    return pd.DataFrame(rows)


def get_entropy_trajectory(
    bundle: ModelBundle,
    sentence: str,
) -> pd.DataFrame:
    """Per-layer entropy of next-token distribution."""
    _, states = _forward_hidden_states(bundle, sentence)
    head = _lm_head(bundle)
    rows: list[dict[str, float | int]] = []
    for i, state in enumerate(states):
        logits = head(state[0, -1, :])
        probs = F.softmax(logits, dim=-1)
        rows.append({"layer": i, "entropy": shannon_entropy(probs)})
    return pd.DataFrame(rows)
