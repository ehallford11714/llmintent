"""Offline smoke test for optional intentisolates soft import."""

from __future__ import annotations

import importlib.util

import pytest


def test_isolates_module_requires_optional_dep():
    spec = importlib.util.find_spec("intentisolates")
    if spec is not None:
        from llmintent import isolates

        assert hasattr(isolates, "identify_isolates")
        return
    with pytest.raises(ImportError, match="intentisolates"):
        importlib.import_module("llmintent.isolates")
