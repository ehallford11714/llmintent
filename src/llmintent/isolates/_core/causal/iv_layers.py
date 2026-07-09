"""Indication (association) vs causation (IV) for layer motifs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from llmintent.isolates._core.causal.features import MotifFeatureTable, pearson


@dataclass
class IndicationScore:
    """Layer / motif association with outcome (not causal)."""

    source: str
    layer: int
    target: str
    association: float
    abs_association: float
    n_obs: int
    method: str = "pearson"
    kind: str = "indication"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalEdgeEstimate:
    """IV / 2SLS edge: endogenous motif/isolate → outcome, instrumented by Z."""

    source: str
    target: str
    instrument: str
    beta_iv: float
    se: float
    first_stage_f: float
    pvalue: float | None
    confidence: float
    layer_x: int
    layer_z: int
    method: str
    weak_instrument: bool = False
    n_obs: int = 0
    kind: str = "causation"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_indication(
    table: MotifFeatureTable,
    *,
    min_abs: float = 0.05,
) -> list[IndicationScore]:
    """Pearson association of each feature column with Y (layer indication)."""
    if not table.rows:
        return []
    y_col = table.outcome_column
    ys = [float(r[y_col]) for r in table.rows]
    scores: list[IndicationScore] = []
    for col in table.columns:
        if col == y_col:
            continue
        xs = [float(r[col]) for r in table.rows]
        r = pearson(xs, ys)
        if abs(r) < min_abs:
            continue
        layer = int(table.column_meta.get(col, {}).get("layer", 2))
        scores.append(
            IndicationScore(
                source=col,
                layer=layer,
                target=y_col,
                association=round(r, 4),
                abs_association=round(abs(r), 4),
                n_obs=len(ys),
            )
        )
    scores.sort(key=lambda s: -s.abs_association)
    return scores


def estimate_layer_iv(
    table: MotifFeatureTable,
    *,
    max_instruments: int = 3,
    max_endogenous: int = 4,
    weak_f_threshold: float = 10.0,
    mock: bool = False,
) -> tuple[list[CausalEdgeEstimate], list[str]]:
    """
    Treat lower-layer features as candidate instruments Z for mid/late
    endogenous features X affecting outcome Y.

    Prefer ``causaliv.estimate_2sls`` when installed; else AutoCausal
    ``_numpy_2sls``; else pure-stdlib Wald IV.
    """
    notes: list[str] = []
    if not table.rows or len(table.rows) < 8:
        notes.append("IV skipped: need ≥8 rows.")
        return [], notes

    y_col = table.outcome_column
    by_layer: dict[int, list[str]] = {}
    for col in table.columns:
        if col == y_col:
            continue
        layer = int(table.column_meta.get(col, {}).get("layer", 2))
        by_layer.setdefault(layer, []).append(col)

    if not by_layer:
        notes.append("IV skipped: no feature columns.")
        return [], notes

    layers_sorted = sorted(by_layer)
    instruments: list[tuple[str, int]] = []
    endogenous: list[tuple[str, int]] = []
    for L in layers_sorted:
        for col in by_layer[L]:
            if L <= 1:
                instruments.append((col, L))
            elif L >= 2:
                endogenous.append((col, L))

    if not instruments and len(layers_sorted) >= 2:
        lo, hi = layers_sorted[0], layers_sorted[-1]
        instruments = [(c, lo) for c in by_layer[lo]]
        endogenous = [(c, hi) for c in by_layer[hi] if hi != lo]
        notes.append(
            f"No L0–L1 features; using earliest layer L{lo} as Z and L{hi} as X."
        )
    if not instruments or not endogenous:
        cols = [c for c in table.columns if c != y_col]
        if len(cols) >= 2:
            mid = max(1, len(cols) // 3)
            instruments = [
                (c, int(table.column_meta.get(c, {}).get("layer", 0))) for c in cols[:mid]
            ]
            endogenous = [
                (c, int(table.column_meta.get(c, {}).get("layer", 2))) for c in cols[mid:]
            ]
            notes.append("Layer-split IV unavailable; split columns into Z/X by name order.")
        else:
            notes.append("IV skipped: need distinct instrument and endogenous columns.")
            return [], notes

    instruments = instruments[:max_instruments]
    endogenous = endogenous[:max_endogenous]

    if mock:
        edges = _mock_iv_edges(instruments, endogenous, y_col, table)
        notes.append("IV edges from mock estimator (tests / offline stub).")
        return edges, notes

    edges: list[CausalEdgeEstimate] = []
    method_used: set[str] = set()

    for z_col, z_layer in instruments:
        for x_col, x_layer in endogenous:
            if z_col == x_col:
                continue
            if x_layer < z_layer:
                continue
            est = _estimate_one_iv(
                table, z_col, x_col, y_col, z_layer, x_layer, weak_f_threshold
            )
            if est is None:
                continue
            # Drop numerically unstable Wald ratios (near-zero first stage, huge beta)
            if est.first_stage_f < 0.05 and abs(est.beta_iv) > 5.0:
                continue
            method_used.add(est.method)
            edges.append(est)

    edges.sort(key=lambda e: (-abs(e.beta_iv), -e.first_stage_f))
    if "causaliv" in method_used:
        notes.append("IV via causaliv.estimate_2sls.")
    elif "autocausal_numpy_2sls" in method_used:
        notes.append("IV via AutoCausal numpy 2SLS lite (causaliv not installed).")
    elif "stdlib_wald_iv" in method_used:
        notes.append("IV via stdlib Wald ratio (causaliv/autocausal IV unavailable).")
    if not edges:
        notes.append("No IV edges estimated (weak first stage or singular design).")
    return edges, notes


def _estimate_one_iv(
    table: MotifFeatureTable,
    z_col: str,
    x_col: str,
    y_col: str,
    z_layer: int,
    x_layer: int,
    weak_f: float,
) -> CausalEdgeEstimate | None:
    zs = [float(r[z_col]) for r in table.rows]
    xs = [float(r[x_col]) for r in table.rows]
    ys = [float(r[y_col]) for r in table.rows]
    n = len(ys)

    try:
        from causaliv import estimate_2sls  # type: ignore

        res = estimate_2sls(
            zs,
            xs,
            ys,
            instrument=z_col,
            endogenous=x_col,
            outcome=y_col,
            weak_f_threshold=weak_f,
        )
        se = float(res.se) if res.se == res.se else float("nan")
        return CausalEdgeEstimate(
            source=x_col,
            target=y_col,
            instrument=z_col,
            beta_iv=float(res.beta_iv),
            se=se,
            first_stage_f=float(res.first_stage_f),
            pvalue=_approx_pvalue(float(res.beta_iv), se),
            confidence=float(res.confidence),
            layer_x=x_layer,
            layer_z=z_layer,
            method="causaliv",
            weak_instrument=bool(res.weak_instrument),
            n_obs=int(res.n_obs),
        )
    except Exception:
        pass

    try:
        from autocausal.iv import _numpy_2sls  # type: ignore
        import numpy as np

        res = _numpy_2sls(
            np.asarray(ys, dtype=float),
            np.asarray(xs, dtype=float),
            np.asarray(zs, dtype=float),
        )
        coef = float(res["coef"])
        se = float(res.get("se", float("nan")))
        f_stat = float(res.get("first_stage_f", 0.0))
        return CausalEdgeEstimate(
            source=x_col,
            target=y_col,
            instrument=z_col,
            beta_iv=round(coef, 6),
            se=round(se, 6) if se == se else float("nan"),
            first_stage_f=round(f_stat, 4),
            pvalue=float(res.get("pvalue", 1.0)),
            confidence=float(
                min(0.95, abs(coef) / (1.0 + abs(se if se == se else 1.0)) * 0.5 + 0.1)
            ),
            layer_x=x_layer,
            layer_z=z_layer,
            method="autocausal_numpy_2sls",
            weak_instrument=f_stat < weak_f,
            n_obs=n,
        )
    except Exception:
        pass

    return _stdlib_wald_iv(
        zs, xs, ys, z_col, x_col, y_col, z_layer, x_layer, weak_f
    )


def _stdlib_wald_iv(
    zs: Sequence[float],
    xs: Sequence[float],
    ys: Sequence[float],
    z_col: str,
    x_col: str,
    y_col: str,
    z_layer: int,
    x_layer: int,
    weak_f: float,
) -> CausalEdgeEstimate | None:
    n = len(zs)
    if n < 5:
        return None
    mz, mx, my = sum(zs) / n, sum(xs) / n, sum(ys) / n
    cov_zx = sum((z - mz) * (x - mx) for z, x in zip(zs, xs)) / (n - 1)
    cov_zy = sum((z - mz) * (y - my) for z, y in zip(zs, ys)) / (n - 1)
    var_z = sum((z - mz) ** 2 for z in zs) / (n - 1)
    if abs(cov_zx) < 1e-10 or var_z < 1e-12:
        return None
    beta = cov_zy / cov_zx
    pi = cov_zx / var_z
    x_hat = [pi * (z - mz) for z in zs]
    ss_model = sum(h**2 for h in x_hat)
    ss_res = sum(((x - mx) - h) ** 2 for x, h in zip(xs, x_hat))
    f_stat = (ss_model / 1) / (ss_res / max(n - 2, 1)) if ss_res > 0 else 0.0
    resid = [(y - my) - beta * (x - mx) for x, y in zip(xs, ys)]
    s2 = sum(r**2 for r in resid) / max(n - 2, 1)
    denom2 = sum(h**2 for h in x_hat)
    se = math.sqrt(s2 / denom2) if denom2 > 1e-12 else float("nan")
    conf = float(min(0.95, abs(cov_zx) * 5.0 + min(0.4, f_stat / 50.0)))
    return CausalEdgeEstimate(
        source=x_col,
        target=y_col,
        instrument=z_col,
        beta_iv=round(float(beta), 6),
        se=round(float(se), 6) if se == se else float("nan"),
        first_stage_f=round(float(f_stat), 4),
        pvalue=_approx_pvalue(float(beta), float(se) if se == se else float("nan")),
        confidence=round(conf, 4),
        layer_x=x_layer,
        layer_z=z_layer,
        method="stdlib_wald_iv",
        weak_instrument=f_stat < weak_f,
        n_obs=n,
    )


def _approx_pvalue(coef: float, se: float) -> float | None:
    if se != se or se <= 0:
        return None
    t = abs(coef) / se
    from math import erfc, sqrt

    return float(erfc(t / sqrt(2.0)))


def _mock_iv_edges(
    instruments: Sequence[tuple[str, int]],
    endogenous: Sequence[tuple[str, int]],
    y_col: str,
    table: MotifFeatureTable,
) -> list[CausalEdgeEstimate]:
    edges: list[CausalEdgeEstimate] = []
    for z_col, z_layer in instruments[:2]:
        for x_col, x_layer in endogenous[:2]:
            if z_col == x_col:
                continue
            zs = [float(r[z_col]) for r in table.rows]
            xs = [float(r[x_col]) for r in table.rows]
            ys = [float(r[y_col]) for r in table.rows]
            r_zx = pearson(zs, xs)
            r_zy = pearson(zs, ys)
            beta = (r_zy / r_zx) if abs(r_zx) > 0.05 else 0.0
            edges.append(
                CausalEdgeEstimate(
                    source=x_col,
                    target=y_col,
                    instrument=z_col,
                    beta_iv=round(beta * 0.5, 4),
                    se=0.1,
                    first_stage_f=round(abs(r_zx) * 40, 2),
                    pvalue=0.05,
                    confidence=0.4,
                    layer_x=x_layer,
                    layer_z=z_layer,
                    method="mock_iv",
                    weak_instrument=abs(r_zx) < 0.2,
                    n_obs=len(ys),
                    notes=["mock"],
                )
            )
    return edges


def layer_indication_matrix(
    indications: Sequence[IndicationScore],
) -> dict[int, float]:
    """Aggregate max |association| per layer."""
    out: dict[int, float] = {}
    for s in indications:
        out[s.layer] = max(out.get(s.layer, 0.0), s.abs_association)
    return dict(sorted(out.items()))


def layer_causation_matrix(
    edges: Sequence[CausalEdgeEstimate],
) -> dict[int, float]:
    """Aggregate max |β_IV| per endogenous layer."""
    out: dict[int, float] = {}
    for e in edges:
        out[e.layer_x] = max(out.get(e.layer_x, 0.0), abs(e.beta_iv))
    return dict(sorted(out.items()))
