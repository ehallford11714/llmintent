"""Tests for LLMIntent Live suite (no model download)."""

from llmintent.live.registry import get_live_model, list_live_models
from llmintent.live.types import LiveAnalyzeResult, LiveModelSpec


def test_list_live_models_includes_qwen_phi():
    keys = {m["key"] for m in list_live_models()}
    assert "qwen-0.5b" in keys
    assert "phi3-mini" in keys
    assert "gpt2" in keys


def test_get_live_model_custom_hf_id():
    spec = get_live_model("org/custom-model")
    assert spec.hf_name == "org/custom-model"
    assert spec.key == "org/custom-model"


def test_live_model_spec_to_dict():
    spec = LiveModelSpec(
        key="test",
        hf_name="test/model",
        params_m=1.0,
        description="test",
        chat_template=True,
    )
    d = spec.to_dict()
    assert d["key"] == "test"
    assert d["chat_template"] is True


def test_analyze_result_to_dict():
    r = LiveAnalyzeResult(
        model_key="gpt2",
        prompt="hello",
        activation_layers={"inference_pivot": 3},
        focus_score=0.5,
        needs_retrace=False,
    )
    d = r.to_dict()
    assert d["focus_score"] == 0.5
    assert d["activation_layers"]["inference_pivot"] == 3
