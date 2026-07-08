"""Ablation conditions for focused reasoning benchmarks."""

from __future__ import annotations

from enum import Enum


class AblationCondition(str, Enum):
    """Benchmark intervention conditions."""

    BASELINE = "baseline"
    FOCUSED = "focused"
    RETRACE = "retrace"
    EXTREME_RETRACE = "extreme_retrace"
    RETRACE_STEER = "retrace_steer"
    EXTREME_STEER = "extreme_steer"
    ITERATIVE = "iterative_heighten"
    EXTREME_ITERATIVE = "extreme_iterative"


DEFAULT_ABLATION_SUITE: tuple[AblationCondition, ...] = (
    AblationCondition.BASELINE,
    AblationCondition.FOCUSED,
    AblationCondition.RETRACE,
    AblationCondition.EXTREME_RETRACE,
    AblationCondition.RETRACE_STEER,
    AblationCondition.EXTREME_STEER,
)

FAST_ABLATION_SUITE: tuple[AblationCondition, ...] = (
    AblationCondition.BASELINE,
    AblationCondition.RETRACE,
    AblationCondition.EXTREME_RETRACE,
)


def parse_conditions(spec: str | list[str] | None) -> list[AblationCondition]:
    if spec is None:
        return list(DEFAULT_ABLATION_SUITE)
    if isinstance(spec, str):
        if spec.lower() in ("default", "all"):
            return list(DEFAULT_ABLATION_SUITE)
        if spec.lower() == "fast":
            return list(FAST_ABLATION_SUITE)
        names = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        names = spec
    out: list[AblationCondition] = []
    for name in names:
        out.append(AblationCondition(name))
    return out
