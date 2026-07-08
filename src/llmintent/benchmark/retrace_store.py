"""Persistent storage for forced retracements and benchmark results."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


@dataclass
class StoredRetracement:
    """A single forced retrace record."""

    id: str
    model_name: str
    benchmark: str
    example_id: str
    condition: str
    context: str
    retrace_prompt: str
    retrace_chain: list[str]
    concepts: list[str]
    focus_baseline: float | None
    focus_after: float | None
    focus_gain: float | None
    predicted: int | None
    label: int | None
    correct: bool | None
    log_probs: list[float] = field(default_factory=list)
    steering_applied: bool = False
    ablation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StoredRetracement:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RetraceStore:
    """Append-only JSONL store for retracements and benchmark outcomes."""

    def __init__(self, path: str = "llmintent_retraces/store.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: StoredRetracement) -> StoredRetracement:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def save_retracement(
        self,
        *,
        model_name: str,
        benchmark: str,
        example_id: str,
        condition: str,
        context: str,
        retrace_prompt: str,
        retrace_chain: list[str] | None = None,
        concepts: list[str] | None = None,
        focus_baseline: float | None = None,
        focus_after: float | None = None,
        predicted: int | None = None,
        label: int | None = None,
        correct: bool | None = None,
        log_probs: list[float] | None = None,
        steering_applied: bool = False,
        ablation: str | None = None,
        metadata: dict | None = None,
    ) -> StoredRetracement:
        gain = None
        if focus_baseline is not None and focus_after is not None:
            gain = focus_after - focus_baseline
        record = StoredRetracement(
            id=str(uuid.uuid4()),
            model_name=model_name,
            benchmark=benchmark,
            example_id=example_id,
            condition=condition,
            context=context,
            retrace_prompt=retrace_prompt,
            retrace_chain=retrace_chain or [retrace_prompt],
            concepts=concepts or [],
            focus_baseline=focus_baseline,
            focus_after=focus_after,
            focus_gain=gain,
            predicted=predicted,
            label=label,
            correct=correct,
            log_probs=log_probs or [],
            steering_applied=steering_applied,
            ablation=ablation,
            metadata=metadata or {},
        )
        return self.append(record)

    def iter_records(self) -> Iterator[StoredRetracement]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield StoredRetracement.from_dict(json.loads(line))

    def to_dataframe(self) -> pd.DataFrame:
        rows = [r.to_dict() for r in self.iter_records()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def summarize_accuracy(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty or "correct" not in df.columns:
            return pd.DataFrame()
        valid = df[df["correct"].notna()]
        if valid.empty:
            return pd.DataFrame()
        return (
            valid.groupby(["model_name", "condition"], dropna=False)
            .agg(
                accuracy=("correct", "mean"),
                n=("correct", "count"),
                mean_focus_gain=("focus_gain", "mean"),
            )
            .reset_index()
            .sort_values(["model_name", "accuracy"], ascending=[True, False])
        )

    def export_csv(self, csv_path: str) -> str:
        df = self.to_dataframe()
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return str(out)
