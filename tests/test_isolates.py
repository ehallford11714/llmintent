"""Offline smoke test for suite isolates (vendored or external)."""

from __future__ import annotations


def test_isolates_module_always_available():
    from llmintent import isolates

    assert hasattr(isolates, "identify_isolates")
    assert isolates.backend_source in ("vendored", "intentisolates")
    isos = isolates.identify_isolates(text="I want X. I cannot Y.")
    assert isos
