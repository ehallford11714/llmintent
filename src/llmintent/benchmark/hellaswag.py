"""HellaSwag benchmark loading and SLM scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llmintent.models import ModelBundle


@dataclass
class HellaSwagExample:
    example_id: str
    context: str
    endings: list[str]
    label: int


def load_hellaswag(
    *,
    split: str = "validation",
    limit: int | None = None,
) -> list[HellaSwagExample]:
    """
    Load HellaSwag examples from HuggingFace datasets.

    Requires optional dependency: pip install llmintent[benchmark]
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HellaSwag loading requires datasets. Install with: pip install llmintent[benchmark]"
        ) from exc

    ds = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
    examples: list[HellaSwagExample] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        ctx = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])
        examples.append(
            HellaSwagExample(
                example_id=str(row.get("ind", i)),
                context=ctx,
                endings=endings,
                label=label,
            )
        )
    return examples


def load_hellaswag_fallback(limit: int = 8) -> list[HellaSwagExample]:
    """Tiny in-memory subset when datasets is unavailable (tests / smoke)."""
    fixtures = [
        HellaSwagExample(
            example_id="0",
            context="A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She",
            endings=[
                "rinses the bucket off with soap and blow dries the dog.",
                "uses a hose to keep it from getting soapy.",
                "gets the dog wet, then it runs away again.",
                "gets into the bucket.",
            ],
            label=2,
        ),
        HellaSwagExample(
            example_id="1",
            context="A man is at a bar and he is drinking a beer. He",
            endings=[
                "throws the beer on the ground.",
                "orders another beer from the bartender.",
                "starts dancing on the table.",
                "leaves the bar and goes for a run.",
            ],
            label=1,
        ),
    ]
    return fixtures[:limit]


def ending_log_prob(
    bundle: ModelBundle,
    prefix: str,
    ending: str,
) -> float:
    """
    Average token log-probability of ending given prefix (causal LM).

    Used for HellaSwag multiple-choice scoring on SLMs.
    """
    tokenizer = bundle.tokenizer
    device = bundle.device

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full = prefix + ending
    enc = tokenizer(full, return_tensors="pt").to(device)
    input_ids = enc.input_ids

    prefix_len = len(prefix_ids)
    if input_ids.shape[1] <= prefix_len:
        return float("-inf")

    with torch.no_grad():
        outputs = bundle.model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_log_probs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)

    start = max(prefix_len - 1, 0)
    ending_log_probs = token_log_probs[0, start:]
    if ending_log_probs.numel() == 0:
        return float("-inf")
    return float(ending_log_probs.mean().item())


def score_hellaswag_example(
    bundle: ModelBundle,
    example: HellaSwagExample,
    *,
    prefix: str | None = None,
) -> tuple[int, list[float], bool]:
    """
    Score one HellaSwag example; return (predicted_index, log_probs, correct).
    """
    ctx = prefix if prefix is not None else example.context
    log_probs = [ending_log_prob(bundle, ctx, end) for end in example.endings]
    predicted = int(max(range(len(log_probs)), key=lambda i: log_probs[i]))
    correct = predicted == example.label
    return predicted, log_probs, correct


def hellaswag_accuracy(
    bundle: ModelBundle,
    examples: list[HellaSwagExample],
    *,
    prefix_fn=None,
) -> dict:
    """Run HellaSwag accuracy over examples."""
    correct = 0
    for ex in examples:
        prefix = prefix_fn(ex) if prefix_fn else None
        _, _, ok = score_hellaswag_example(bundle, ex, prefix=prefix)
        if ok:
            correct += 1
    n = len(examples)
    return {
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "total": n,
    }
