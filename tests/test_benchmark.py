"""Benchmark tests (no full model load)."""

import json
import tempfile
from pathlib import Path

from llmintent.benchmark.ablation import AblationCondition, parse_conditions
from llmintent.benchmark.hellaswag import load_hellaswag_fallback
from llmintent.benchmark.retrace_store import RetraceStore
from llmintent.benchmark.runner import _build_prefix, _infer_concepts
from llmintent.heighten.extreme import ExtremeRetraceMode, build_extreme_retrace_chain


def test_parse_conditions_fast():
    conds = parse_conditions("fast")
    assert AblationCondition.BASELINE in conds
    assert AblationCondition.EXTREME_RETRACE in conds


def test_retrace_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "store.jsonl")
        store = RetraceStore(path)
        store.save_retracement(
            model_name="gpt2",
            benchmark="hellaswag",
            example_id="0",
            condition="retrace",
            context="A man is walking",
            retrace_prompt="A man is walking\nWait — let me retrace",
            retrace_chain=["pass1", "pass2"],
            concepts=["walking"],
            focus_baseline=0.3,
            focus_after=0.6,
            predicted=1,
            label=1,
            correct=True,
            log_probs=[-1.0, -0.5, -2.0, -3.0],
        )
        df = store.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["correct"] == True
        summary = store.summarize_accuracy()
        assert summary.iloc[0]["accuracy"] == 1.0


def test_extreme_retrace_chain_has_multiple_passes():
    chain = build_extreme_retrace_chain(
        "Question: context here. She",
        concepts=["context", "commonsense"],
        mode=ExtremeRetraceMode.TRIPLE.value,
    )
    assert len(chain.passes) >= 2
    assert "STRICT FOCUS MODE" in chain.combined_prompt
    assert "retrace" in chain.combined_prompt.lower()


def test_build_prefix_extreme():
    ex = load_hellaswag_fallback(limit=1)[0]
    prefix, rp, chain = _build_prefix(
        ex,
        AblationCondition.EXTREME_RETRACE,
        extreme_mode=ExtremeRetraceMode.TRIPLE.value,
    )
    assert len(prefix) > len(ex.context)
    assert len(chain) >= 1


def test_infer_concepts():
    concepts = _infer_concepts("A woman is outside with a bucket and a dog.")
    assert len(concepts) <= 5
    assert any("woman" in c or "bucket" in c or "dog" in c for c in concepts)
