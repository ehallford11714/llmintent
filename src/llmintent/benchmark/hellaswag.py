"""HellaSwag benchmark loading and SLM scoring."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from llmintent.models import ModelBundle

DEFAULT_HELLASWAG_VAL_URL = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
)
DEFAULT_HELLASWAG_VAL_PATH = Path(__file__).resolve().parents[3] / "data" / "hellaswag_val.jsonl"


@dataclass
class HellaSwagExample:
    example_id: str
    context: str
    endings: list[str]
    label: int


def ensure_hellaswag_val_jsonl(path: Path | None = None) -> Path:
    """Download validation JSONL if missing."""
    path = path or DEFAULT_HELLASWAG_VAL_PATH
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DEFAULT_HELLASWAG_VAL_URL, path)
    return path


def load_hellaswag_jsonl(
    *,
    path: Path | str | None = None,
    limit: int | None = None,
) -> list[HellaSwagExample]:
    """Load HellaSwag validation examples from local JSONL (official release)."""
    path = Path(path) if path is not None else ensure_hellaswag_val_jsonl()
    examples: list[HellaSwagExample] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            examples.append(
                HellaSwagExample(
                    example_id=str(row.get("ind", i)),
                    context=row["ctx"],
                    endings=row["endings"],
                    label=int(row["label"]),
                )
            )
    return examples


def load_hellaswag(
    *,
    split: str = "validation",
    limit: int | None = None,
) -> list[HellaSwagExample]:
    """
    Load HellaSwag examples.

    Prefers local official JSONL; falls back to HuggingFace `datasets`.
    """
    if split not in ("validation", "val"):
        raise ValueError(f"Only validation split is supported locally, got {split!r}")

    if DEFAULT_HELLASWAG_VAL_PATH.exists():
        return load_hellaswag_jsonl(path=DEFAULT_HELLASWAG_VAL_PATH, limit=limit)

    try:
        from datasets import load_dataset
    except ImportError:
        return load_hellaswag_jsonl(limit=limit)

    try:
        ds = load_dataset("Rowan/hellaswag", split="validation")
        examples: list[HellaSwagExample] = []
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            examples.append(
                HellaSwagExample(
                    example_id=str(row.get("ind", i)),
                    context=row["ctx"],
                    endings=row["endings"],
                    label=int(row["label"]),
                )
            )
        return examples
    except Exception:
        return load_hellaswag_jsonl(limit=limit)


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
