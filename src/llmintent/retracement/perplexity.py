"""Perplexity evaluation for Retracement Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from llmintent.models import ModelBundle, load_model_bundle
from llmintent.retracement.config import RetracementConfig, RetracementMode
from llmintent.retracement.transformer import RetracementTransformer


@dataclass
class PerplexityResult:
    model_name: str
    mode: str
    perplexity: float
    avg_nll: float
    num_tokens: int
    num_sequences: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "perplexity": self.perplexity,
            "avg_nll": self.avg_nll,
            "num_tokens": self.num_tokens,
            "num_sequences": self.num_sequences,
        }


def load_eval_texts(
    *,
    split: str = "validation",
    limit: int = 32,
    max_chars: int = 512,
) -> list[str]:
    """Load WikiText-2 snippets or fallback corpus."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts: list[str] = []
        for row in ds:
            t = str(row["text"]).strip()
            if len(t) < 20:
                continue
            texts.append(t[:max_chars])
            if len(texts) >= limit:
                break
        if texts:
            return texts
    except ImportError:
        pass

    return _fallback_corpus(limit)


def _fallback_corpus(limit: int) -> list[str]:
    base = [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Machine learning models encode statistical patterns from large text corpora.",
        "Retracement reasoning forces the network to reconsider its internal computation path.",
        "Focused reasoning concentrates activation in workspace layers before motor commit.",
        "Language models predict the next token using contextual hidden representations.",
        "Commonsense reasoning requires binding entities and relations across depth.",
        "Transformers process sequences through stacked self-attention and feedforward blocks.",
        "Perplexity measures how well a model predicts held-out text under cross entropy.",
    ]
    out: list[str] = []
    i = 0
    while len(out) < limit:
        out.append(base[i % len(base)])
        i += 1
    return out


def compute_perplexity(
    retracement: RetracementTransformer,
    texts: list[str],
    *,
    max_length: int = 128,
) -> PerplexityResult:
    """Compute perplexity over text snippets."""
    bundle = retracement.bundle
    tokenizer = bundle.tokenizer
    device = bundle.device

    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        input_ids = enc.input_ids
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            nll, n_tok = retracement.token_loss(input_ids)
        total_nll += float(nll.item())
        total_tokens += n_tok

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 20.0))

    return PerplexityResult(
        model_name=bundle.name,
        mode=retracement.config.mode.value,
        perplexity=ppl,
        avg_nll=avg_nll,
        num_tokens=total_tokens,
        num_sequences=len(texts),
    )


def evaluate_perplexity(
    model_name: str,
    config: RetracementConfig,
    texts: list[str] | None = None,
    *,
    limit: int = 32,
) -> PerplexityResult:
    """Load model, wrap with RetracementTransformer, compute perplexity."""
    bundle = load_model_bundle(model_name)
    texts = texts or load_eval_texts(limit=limit)
    rt = RetracementTransformer(bundle, config)
    try:
        return compute_perplexity(rt, texts)
    finally:
        del bundle.model
