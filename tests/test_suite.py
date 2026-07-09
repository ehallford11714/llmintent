"""Offline tests for the curated model suite registry (no multi-GB downloads)."""

from __future__ import annotations

import os

import pytest

from llmintent.suite import (
    FAMILIES,
    SIZES,
    get_model_spec,
    list_models,
    resolve_model_id,
)
from llmintent.suite.loader import should_run_load_tests
from llmintent.suite.resolve import resolve_from_env, resolve_model_spec


def test_families_and_sizes():
    assert set(FAMILIES) >= {"qwen", "mistral", "minimax", "glm", "legacy"}
    assert list(SIZES) == ["tiny", "small", "medium", "large", "xl"]


def test_list_models_offline():
    rows = list_models()
    assert len(rows) >= 20
    keys = {r["key"] for r in rows}
    assert "qwen:medium" in keys
    assert "mistral:small" in keys
    assert "minimax:xl" in keys
    assert "glm:small" in keys
    assert "legacy:tiny" in keys


def test_list_models_by_family():
    qwen = list_models(family="qwen")
    assert all(r["family"] == "qwen" for r in qwen)
    assert len(qwen) == 5


def test_get_model_spec_ids():
    assert get_model_spec("qwen", "tiny").hf_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert get_model_spec("qwen", "medium").hf_id == "Qwen/Qwen2.5-7B-Instruct"
    assert get_model_spec("mistral", "tiny").hf_id.startswith("mistralai/")
    assert get_model_spec("minimax", "medium").hf_id == "MiniMaxAI/MiniMax-M2"
    assert get_model_spec("glm", "small").hf_id in {
        "zai-org/GLM-4-9B-0414",
        "THUDM/GLM-4-9B-0414",
    }
    assert get_model_spec("legacy", "small").hf_id == "gpt2"
    assert get_model_spec("legacy", "tiny").hf_id == "distilgpt2"


def test_aliases():
    assert get_model_spec("Qwen3", "md").hf_id == get_model_spec("qwen", "medium").hf_id
    assert get_model_spec("ministral", "sm").family == "mistral"
    assert get_model_spec("zhipu", "small").family == "glm"


def test_resolve_model_id_family_size():
    assert resolve_model_id(family="qwen", size="small") == "Qwen/Qwen2.5-3B-Instruct"
    assert resolve_model_id(model="qwen:large") == "Qwen/Qwen2.5-32B-Instruct"
    assert resolve_model_id(model="gpt2") == "gpt2"
    assert resolve_model_id(default="distilgpt2") == "distilgpt2"


def test_resolve_from_env(monkeypatch):
    monkeypatch.setenv("LLMINTENT_FAMILY", "mistral")
    monkeypatch.setenv("LLMINTENT_SIZE", "tiny")
    monkeypatch.delenv("LLMINTENT_MODEL", raising=False)
    assert resolve_model_id(use_env=True).startswith("mistralai/")
    info = resolve_from_env()
    assert info["family"] == "mistral"
    assert "Ministral" in (info["resolved_id"] or "") or "mistral" in (info["resolved_id"] or "").lower()


def test_resolve_env_model_key(monkeypatch):
    monkeypatch.setenv("LLMINTENT_MODEL", "glm:medium")
    monkeypatch.delenv("LLMINTENT_FAMILY", raising=False)
    assert "GLM" in resolve_model_id(use_env=True) or "glm" in resolve_model_id(use_env=True).lower()


def test_unknown_family():
    with pytest.raises(KeyError):
        get_model_spec("notafamily", "medium")


def test_suite_key_spec():
    spec = resolve_model_spec(model="qwen:tiny")
    assert spec is not None
    assert spec.hf_id == "Qwen/Qwen2.5-0.5B-Instruct"


def test_analyzer_from_suite_resolves():
    from llmintent.suite import resolve_model_id

    assert resolve_model_id(family="legacy", size="tiny") == "distilgpt2"


@pytest.mark.skipif(not should_run_load_tests(), reason="Set LLMINTENT_LOAD_TEST=1 to download weights")
def test_load_legacy_tiny():
    from llmintent.suite import load_suite_model

    bundle = load_suite_model(family="legacy", size="tiny", device="cpu")
    assert bundle.name == "distilgpt2"
    assert bundle.num_layers > 0
