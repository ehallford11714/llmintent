"""Dense weight extraction for SVD.

Packed 4-bit storage must be dequantized. Casting ``weight.data`` with
``.float()`` reconstructs neither the logical matrix nor the right shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np

QUANT_MARKERS = ("params4bit", "nf4", "linear4bit", "linear8bit", "int8params")


class WeightExtractionError(ValueError):
    """Quantized or malformed weights that must not be fed to SVD."""


def get_module(model: Any, path: str) -> Any:
    """Resolve ``model.layers.0.mlp.down_proj.weight`` to the Linear module."""
    obj = model
    parts = path.replace(".weight", "").split(".")
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def looks_quantized(obj: Any) -> bool:
    if obj is None:
        return False
    if getattr(obj, "quant_state", None) is not None:
        return True
    name = type(obj).__name__.lower()
    if any(marker in name for marker in QUANT_MARKERS):
        return True
    weight = getattr(obj, "weight", None)
    if weight is not None and weight is not obj:
        return looks_quantized(weight)
    return False


def logical_linear_shape(module: Any) -> tuple[int, int] | None:
    out_f = getattr(module, "out_features", None)
    in_f = getattr(module, "in_features", None)
    if out_f and in_f:
        return (int(out_f), int(in_f))
    return None


def extract_dense_weight(
    obj: Any,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return a dense float matrix. Never cast packed NF4 storage."""
    if obj is None:
        raise WeightExtractionError("no weight object")
    if isinstance(obj, np.ndarray):
        return _assert_matrix(np.asarray(obj, dtype=np.float64), expected_shape)

    dequantized = _try_dequantize(obj)
    if dequantized is not None:
        return _assert_matrix(_to_numpy(dequantized), expected_shape)

    if looks_quantized(obj):
        raise WeightExtractionError(
            "quantized weights require bitsandbytes dequantize_4bit or "
            "Linear4bit.dequantize(); casting packed storage with .float() is invalid"
        )

    weight = getattr(obj, "weight", obj)
    if looks_quantized(weight):
        raise WeightExtractionError(
            "quantized Parameter requires explicit dequantization before SVD"
        )
    tensor = getattr(weight, "data", weight)
    if looks_quantized(tensor):
        raise WeightExtractionError(
            "packed weight.data is not the effective matrix; dequantize first"
        )
    return _assert_matrix(_to_numpy(tensor), expected_shape)


def _try_dequantize(obj: Any) -> Any | None:
    dequant = getattr(obj, "dequantize", None)
    if callable(dequant):
        try:
            out = dequant()
        except Exception as exc:
            raise WeightExtractionError(f"module.dequantize() failed: {exc}") from exc
        if out is not None and not looks_quantized(out):
            return out
    weight = getattr(obj, "weight", None)
    if weight is not None and weight is not obj:
        got = _try_dequantize(weight)
        if got is not None:
            return got
        dequant_w = getattr(weight, "dequantize", None)
        if callable(dequant_w):
            try:
                out = dequant_w()
            except Exception as exc:
                raise WeightExtractionError(f"weight.dequantize() failed: {exc}") from exc
            if out is not None and not looks_quantized(out):
                return out
    quant_state = getattr(obj, "quant_state", None)
    if quant_state is None and weight is not None:
        quant_state = getattr(weight, "quant_state", None)
        obj = weight
    if quant_state is not None:
        try:
            import bitsandbytes.functional as bnb_f
        except Exception as exc:
            raise WeightExtractionError(
                "NF4 weights need bitsandbytes.functional.dequantize_4bit"
            ) from exc
        try:
            return bnb_f.dequantize_4bit(obj, quant_state=quant_state)
        except Exception as exc:
            raise WeightExtractionError(f"dequantize_4bit failed: {exc}") from exc
    return None


def _to_numpy(tensor: Any) -> np.ndarray:
    value = tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _assert_matrix(dense: np.ndarray, expected: tuple[int, ...] | None) -> np.ndarray:
    dense = np.ascontiguousarray(dense, dtype=np.float64)
    if dense.ndim != 2:
        raise WeightExtractionError(f"expected a matrix, got shape {tuple(dense.shape)}")
    if expected:
        exp = tuple(int(x) for x in expected)
        if dense.shape != exp:
            raise WeightExtractionError(
                f"dequantized shape {tuple(dense.shape)} != logical {exp}"
            )
    return dense
