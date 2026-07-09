"""Build AutoCausal-ready feature tables from isolates and motifs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from llmintent.isolates._core.motifs import _layer_sort_key, _typ
from llmintent.isolates._core.types import Isolate, Motif

_SAFE = re.compile(r"[^a-zA-Z0-9_]+")


def column_name_for_isolate(iso: Isolate) -> str:
    """Stable column: ``isolate_<typology>_L<layer>`` (optionally + short id)."""
    typ = _typ(iso)
    layer = _layer_sort_key(iso.layer)
    base = f"isolate_{typ}_L{layer}"
    return _SAFE.sub("_", base).strip("_").lower()


def column_name_for_motif(motif: Motif) -> str:
    """Stable column: ``motif_<typology>_L<min_layer>`` (+ pattern slug)."""
    typ = motif.typology.value if hasattr(motif.typology, "value") else str(motif.typology)
    if motif.layers:
        layer = min(_layer_sort_key(L) for L in motif.layers)
    else:
        layer = 2
    pat = (motif.pattern or motif.id)[:40]
    slug = _SAFE.sub("_", pat).strip("_").lower()[:28] or "m"
    base = f"motif_{typ}_L{layer}_{slug}"
    return _SAFE.sub("_", base).strip("_").lower()


@dataclass
class MotifFeatureTable:
    """Tabular view of isolate/motif activations + outcome."""

    rows: list[dict[str, float]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    outcome_column: str = "Y"
    column_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_records(self) -> list[dict[str, float]]:
        return [dict(r) for r in self.rows]

    def to_dataframe(self) -> Any:
        """Return a pandas DataFrame when pandas is installed; else list of dicts."""
        try:
            import pandas as pd
        except Exception:
            return self.to_records()
        if not self.rows:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.rows)[self.columns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": len(self.rows),
            "columns": list(self.columns),
            "outcome_column": self.outcome_column,
            "column_meta": dict(self.column_meta),
            "notes": list(self.notes),
            "preview": self.rows[:3],
        }


def build_feature_frame(
    isolates: Sequence[Isolate],
    motifs: Sequence[Motif] | None = None,
    *,
    outcome: Sequence[float] | Mapping[str, float] | float | None = None,
    outcome_column: str = "Y",
    outcome_hint: str | None = None,
    n_bootstrap: int = 48,
    seed: int = 17,
    include_isolates: bool = True,
    include_motifs: bool = True,
) -> MotifFeatureTable:
    """
    Convert isolate/motif activations into a multi-row table for AutoCausal / IV.

    A single text yields one structural activation vector. We expand to
    ``n_bootstrap`` synthetic rows with light noise so association / 2SLS
    estimators have enough observations offline. When ``outcome`` is omitted,
    Y is derived from outcome/action/goal isolate strengths (or ``outcome_hint``).
    """
    motifs = list(motifs or [])
    isolates = list(isolates)
    notes: list[str] = []

    # Aggregate presence/strength by column (collapse same typology@layer)
    strengths: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}

    if include_isolates:
        for iso in isolates:
            col = column_name_for_isolate(iso)
            conf = float(iso.confidence or 0.5)
            # Presence + confidence as strength
            strengths[col] = strengths.get(col, 0.0) + max(0.05, conf)
            meta[col] = {
                "kind": "isolate",
                "typology": _typ(iso),
                "layer": _layer_sort_key(iso.layer),
                "layer_name": iso.layer_name,
                "ids": meta.get(col, {}).get("ids", []) + [iso.id],
            }

    if include_motifs:
        for m in motifs:
            col = column_name_for_motif(m)
            strength = float(m.support or 0.2) * float(m.confidence or 0.5) * 2.0
            strengths[col] = strengths.get(col, 0.0) + max(0.05, strength)
            layers = [_layer_sort_key(L) for L in m.layers] if m.layers else [2]
            meta[col] = {
                "kind": "motif",
                "typology": m.typology.value if hasattr(m.typology, "value") else str(m.typology),
                "layer": min(layers),
                "layers": layers,
                "pattern": m.pattern,
                "ids": meta.get(col, {}).get("ids", []) + [m.id],
            }

    if not strengths:
        notes.append("No isolate/motif features; empty frame.")
        return MotifFeatureTable(
            rows=[],
            columns=[outcome_column],
            outcome_column=outcome_column,
            notes=notes,
        )

    # Cap / normalize strengths into [0, 1]-ish range
    max_s = max(strengths.values()) or 1.0
    base = {c: min(1.0, v / max_s) for c, v in strengths.items()}

    y_base = _resolve_outcome(
        isolates,
        base,
        meta,
        outcome=outcome,
        outcome_hint=outcome_hint,
        notes=notes,
    )

    feature_cols = sorted(base.keys())
    columns = feature_cols + [outcome_column]

    rows = _bootstrap_rows(
        base,
        y_base,
        feature_cols,
        outcome_column=outcome_column,
        n=max(12, int(n_bootstrap)),
        seed=seed,
        meta=meta,
    )
    notes.append(
        f"Expanded 1 activation vector → {len(rows)} bootstrap rows "
        f"(seed={seed}) for offline association/IV."
    )
    notes.append(
        "Bootstrap rows are synthetic perturbations of motif/isolate strengths — "
        "not independent observational units. Treat IV estimates as exploratory."
    )

    return MotifFeatureTable(
        rows=rows,
        columns=columns,
        outcome_column=outcome_column,
        column_meta=meta,
        notes=notes,
    )


def _resolve_outcome(
    isolates: Sequence[Isolate],
    base: Mapping[str, float],
    meta: Mapping[str, dict[str, Any]],
    *,
    outcome: Sequence[float] | Mapping[str, float] | float | None,
    outcome_hint: str | None,
    notes: list[str],
) -> float:
    if isinstance(outcome, (int, float)):
        return float(outcome)
    if isinstance(outcome, Mapping):
        # mean of provided values
        vals = [float(v) for v in outcome.values()]
        return sum(vals) / max(len(vals), 1) if vals else 0.0
    if isinstance(outcome, Sequence) and not isinstance(outcome, (str, bytes)):
        vals = [float(v) for v in outcome]
        return sum(vals) / max(len(vals), 1) if vals else 0.0

    # Derive from outcome/action/goal isolates
    score = 0.0
    weight = 0.0
    hint = (outcome_hint or "").lower().strip()
    for iso in isolates:
        typ = _typ(iso)
        w = 0.0
        if typ == "outcome":
            w = 1.0
        elif typ == "action":
            w = 0.7
        elif typ == "goal":
            w = 0.5
        elif hint and hint in (iso.label or "").lower():
            w = 0.9
        elif hint and hint in typ:
            w = 0.8
        if w > 0:
            score += w * float(iso.confidence or 0.5)
            weight += w
    if weight > 0:
        notes.append("Outcome Y derived from outcome/action/goal isolate strengths.")
        return min(1.0, score / weight)

    # Fallback: late-layer motif/isolate mean
    late = [
        base[c]
        for c, m in meta.items()
        if c in base and int(m.get("layer", 0)) >= 3
    ]
    if late:
        notes.append("Outcome Y fallback: mean of L3+ feature strengths.")
        return sum(late) / len(late)

    notes.append("Outcome Y fallback: mean of all feature strengths.")
    return sum(base.values()) / max(len(base), 1)


def _bootstrap_rows(
    base: Mapping[str, float],
    y_base: float,
    feature_cols: Sequence[str],
    *,
    outcome_column: str,
    n: int,
    seed: int,
    meta: Mapping[str, dict[str, Any]],
) -> list[dict[str, float]]:
    """Deterministic LCG noise around base activations + structured Y link."""
    # Simple LCG — no numpy required
    state = seed & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 0xFFFFFFFF

    # Causal sketch for synthetic Y: lower layers weakly → mid → late → Y
    # so IV (Z=early, X=mid) has a recoverable signal under the generative model.
    layer_of = {c: int(meta.get(c, {}).get("layer", 2)) for c in feature_cols}

    early_cols = [c for c in feature_cols if layer_of[c] <= 1]
    mid_cols = [c for c in feature_cols if layer_of[c] == 2]
    late_cols = [c for c in feature_cols if layer_of[c] >= 3]

    rows: list[dict[str, float]] = []
    for _ in range(n):
        row: dict[str, float] = {}
        # Draw early (Z) first, then mid/late (X) with a relevance path from early
        # so IV first-stage is not vacuously weak under the synthetic DGP.
        for c in early_cols:
            noise = (rnd() - 0.5) * 0.25
            row[c] = max(0.0, min(1.5, base[c] + noise))
        early_mean = (sum(row[c] for c in early_cols) / len(early_cols)) if early_cols else 0.0

        for c in mid_cols:
            noise = (rnd() - 0.5) * 0.2
            row[c] = max(0.0, min(1.5, 0.55 * base[c] + 0.35 * early_mean + noise))
        mid_mean = (sum(row[c] for c in mid_cols) / len(mid_cols)) if mid_cols else early_mean

        for c in late_cols:
            noise = (rnd() - 0.5) * 0.2
            # Late X depends on mid (+ weak early) — exclusion imperfect on purpose
            row[c] = max(
                0.0,
                min(1.5, 0.45 * base[c] + 0.40 * mid_mean + 0.10 * early_mean + noise),
            )

        # Any remaining columns (shouldn't happen) get independent noise
        for c in feature_cols:
            if c not in row:
                row[c] = max(0.0, min(1.5, base[c] + (rnd() - 0.5) * 0.25))

        late = [row[c] for c in late_cols]
        mid = [row[c] for c in mid_cols]
        early = [row[c] for c in early_cols]
        y = y_base
        if late:
            y = 0.35 * y + 0.45 * (sum(late) / len(late))
        if mid:
            y = 0.7 * y + 0.25 * (sum(mid) / len(mid))
        if early:
            # small direct path (confounding) so indication is not pure causation
            y = 0.9 * y + 0.08 * (sum(early) / len(early))
        y = max(0.0, min(1.5, y + (rnd() - 0.5) * 0.12))
        row[outcome_column] = y
        rows.append(row)
    return rows


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation (stdlib)."""
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    x = list(xs)[:n]
    y = list(ys)[:n]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)
