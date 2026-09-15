"""Connectome-guided IV: sensory regions instrument central regions.

Identification prior from fly anatomy:

* Z (instrument) = sensory region occupancy (receptors)
* X (endogenous) = central region occupancy (associative / causal / workspace)
* Y (outcome) = motor / descending occupancy, or an explicit outcome hint

A pair is admitted only if the literature-core graph has a path Z→X.
Sensory→motor short pipes are recorded as exclusion violations and are
not used as instruments for that outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from llmintent.anatomy.atlas import abstract_layer, band_of, region_ids
from llmintent.anatomy.compile import RegionPlan, compile_regions
from llmintent.anatomy.connectome import RegionConnectome, literature_region_connectome
from llmintent.isolates._core.causal.features import MotifFeatureTable
from llmintent.isolates._core.causal.iv_layers import (
    CausalEdgeEstimate,
    IndicationScore,
    estimate_indication,
    estimate_layer_iv,
)


@dataclass
class AnatomyIVResult:
    indications: list[IndicationScore] = field(default_factory=list)
    causation_edges: list[CausalEdgeEstimate] = field(default_factory=list)
    indication_by_region: dict[str, float] = field(default_factory=dict)
    causation_by_region: dict[str, float] = field(default_factory=dict)
    instruments_used: list[str] = field(default_factory=list)
    exclusion_violations: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    occupancy: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "indications": [s.to_dict() for s in self.indications],
            "causation_edges": [e.to_dict() for e in self.causation_edges],
            "indication_by_region": dict(self.indication_by_region),
            "causation_by_region": dict(self.causation_by_region),
            "instruments_used": list(self.instruments_used),
            "exclusion_violations": list(self.exclusion_violations),
            "occupancy": {k: round(v, 4) for k, v in self.occupancy.items()},
            "notes": list(self.notes),
        }


def occupancy_table(
    occupancy: Mapping[str, float],
    *,
    n_bootstrap: int = 48,
    seed: int = 17,
    outcome_region: str = "motor",
) -> MotifFeatureTable:
    """Turn a region occupancy vector into a bootstrap feature table for IV."""
    rng = __import__("random").Random(seed)
    cols = [f"region_{rid}" for rid in region_ids()]
    meta: dict[str, dict[str, Any]] = {}
    for rid in region_ids():
        meta[f"region_{rid}"] = {
            "layer": abstract_layer(rid),
            "region": rid,
            "band": band_of(rid),
        }
    y_col = "Y"
    rows: list[dict[str, float]] = []
    base = {rid: float(occupancy.get(rid, 0.0)) for rid in region_ids()}
    y0 = float(occupancy.get(outcome_region, 0.0))
    for _ in range(max(n_bootstrap, 8)):
        row: dict[str, float] = {}
        for rid in region_ids():
            noise = rng.uniform(-0.08, 0.08)
            row[f"region_{rid}"] = max(0.0, base[rid] + noise)
        # Outcome tracks motor plus a leak from central regions (confounded assoc.)
        leak = 0.15 * row["region_causal_logic"] + 0.1 * row["region_workspace"]
        row[y_col] = max(0.0, y0 + leak + rng.uniform(-0.05, 0.05))
        rows.append(row)
    table = MotifFeatureTable(
        rows=rows,
        columns=cols + [y_col],
        outcome_column=y_col,
        column_meta=meta,
        notes=[
            "Bootstrap rows jitter region occupancy from one compiled/SVD map.",
            "Simulation fixture only — not independent observations, not population inference.",
            "Do not treat resulting IV edges as empirical intervention-supported pathways.",
        ],
    )
    return table


def connectome_iv(
    occupancy: Mapping[str, float],
    *,
    connectome: RegionConnectome | None = None,
    outcome_region: str = "motor",
    n_bootstrap: int = 48,
    seed: int = 17,
    mock_iv: bool = False,
) -> AnatomyIVResult:
    conn = connectome or literature_region_connectome()
    table = occupancy_table(
        occupancy,
        n_bootstrap=n_bootstrap,
        seed=seed,
        outcome_region=outcome_region,
    )
    indications = estimate_indication(table, min_abs=0.03)

    # Restrict instruments to sensory regions with a path into a central X.
    allowed_z: set[str] = set()
    allowed_x: set[str] = set()
    for rid in region_ids():
        if band_of(rid) == "central":
            for z in conn.valid_instruments(rid):
                if conn.has_edge(z, outcome_region) or (
                    band_of(z) == "sensory" and conn.has_edge(z, "descending")
                ):
                    continue
                allowed_z.add(z)
                allowed_x.add(rid)

    # Drop disallowed columns from a copy used for IV.
    keep_cols = [
        c
        for c in table.columns
        if c == table.outcome_column
        or table.column_meta.get(c, {}).get("region") in allowed_z
        or table.column_meta.get(c, {}).get("region") in allowed_x
    ]
    iv_table = MotifFeatureTable(
        rows=[{k: r[k] for k in keep_cols if k in r} for r in table.rows],
        columns=keep_cols,
        outcome_column=table.outcome_column,
        column_meta={k: v for k, v in table.column_meta.items() if k in keep_cols},
        notes=list(table.notes),
    )
    edges, iv_notes = estimate_layer_iv(iv_table, mock=mock_iv)

    ind_by: dict[str, float] = {}
    for s in indications:
        rid = table.column_meta.get(s.source, {}).get("region", s.source)
        ind_by[rid] = max(ind_by.get(rid, 0.0), s.abs_association)
    cau_by: dict[str, float] = {}
    for e in edges:
        rid = table.column_meta.get(e.source, {}).get("region", e.source)
        cau_by[rid] = max(cau_by.get(rid, 0.0), abs(e.beta_iv))

    notes = [
        "Instruments = sensory regions with a literature-core path into central X.",
        "Outcome Y defaults to motor occupancy (token emission analogue).",
        "IV table is a simulation fixture (jittered occupancy), not a designed intervention.",
        *iv_notes,
    ]
    return AnatomyIVResult(
        indications=indications,
        causation_edges=edges,
        indication_by_region=ind_by,
        causation_by_region=cau_by,
        instruments_used=sorted(allowed_z),
        exclusion_violations=[e.to_dict() for e in conn.exclusion_violations()],
        notes=notes,
        occupancy={k: float(v) for k, v in occupancy.items()},
    )


def iv_from_text(
    text: str,
    *,
    occupancy: Mapping[str, float] | None = None,
    mock_iv: bool = False,
    seed: int = 17,
) -> tuple[RegionPlan, AnatomyIVResult]:
    plan = compile_regions(text)
    occ = dict(occupancy) if occupancy is not None else plan.occupancy()
    # Guarantee a little motor mass so Y is not identically zero.
    if occ.get("motor", 0.0) < 0.05 and plan.hits:
        occ["motor"] = 0.2 + 0.3 * occ.get("descending", 0.0)
    result = connectome_iv(occ, mock_iv=mock_iv, seed=seed)
    return plan, result
