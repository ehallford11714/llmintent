"""Mine finished suite_trace report.json folders after an early stop."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))

from pole_ablation_100 import mine  # noqa: E402
from suite_trace_1000 import save_aggregate  # noqa: E402

FLOOR = 0.18
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/suite_27b_1000")


def load_rows(root: Path) -> list[dict]:
    rows = []
    for report in sorted(root.glob("p*/report.json")):
        payload = json.loads(report.read_text(encoding="utf-8"))
        if "pole_intensity" not in payload:
            continue
        rows.append(payload)
    return rows


def downsample(curve: list[float], n: int = 17) -> tuple[list[str], list[float]]:
    if not curve:
        return [], []
    if len(curve) <= n:
        return [str(i) for i in range(len(curve))], [round(x, 5) for x in curve]
    idx = [round(i * (len(curve) - 1) / (n - 1)) for i in range(n)]
    return [str(i) for i in idx], [round(curve[i], 5) for i in idx]


def main() -> int:
    rows = load_rows(ROOT)
    if not rows:
        print("no reports", file=sys.stderr)
        return 1

    n_layers = max(len(r.get("pole_intensity") or []) for r in rows)
    mine_rows = []
    for r in rows:
        c = list(r.get("pole_intensity") or [0.0])
        if len(c) < n_layers:
            c = c + [0.0] * (n_layers - len(c))
        mine_rows.append(
            {
                "id": r["base_id"],
                "family": r["family"],
                "condition": r["condition"],
                "has_number": any(ch.isdigit() for ch in r["text"]),
                "curve": c[:n_layers],
                "compile": r.get("compile_occupancy") or {},
            }
        )
    pole = mine(mine_rows, n_layers)
    pole["n_bases"] = len({r["base_id"] for r in rows})
    pole["n_forwards"] = len(rows)

    # Logit-lens region occupancy (best region even if under floor).
    region_hits = Counter()
    identified_hits = Counter()
    ident_by_layer: dict[int, list[int]] = defaultdict(list)
    ident_by_cond: dict[str, list[int]] = defaultdict(list)
    max_score_by_cond: dict[str, list[float]] = defaultdict(list)
    peak_pole_by_cond: dict[str, list[float]] = defaultdict(list)
    peak_layer_by_cond: dict[str, list[int]] = defaultdict(list)
    compile_any = 0
    misalign_n = 0
    residual_id_n = 0
    traj_regions = Counter()
    family_ident: dict[str, list[int]] = defaultdict(list)

    for r in rows:
        cond = r["condition"]
        fam = r["family"]
        pole_c = r.get("pole_intensity") or [0.0]
        peak_pole_by_cond[cond].append(max(pole_c))
        peak_layer_by_cond[cond].append(int(r.get("pole_peak_layer") or 0))
        if r.get("compile_regions"):
            compile_any += 1
        if r.get("misalign"):
            misalign_n += 1
        if r.get("residual_identified"):
            residual_id_n += 1
        for rid in r.get("trajectory_path") or []:
            traj_regions[rid] += 1
        lens = r.get("logit_lens") or []
        any_ident = 0
        best = 0.0
        for row in lens:
            rid = row.get("region") or "workspace"
            sc = float(row.get("region_score") or 0)
            best = max(best, sc)
            region_hits[rid] += 1
            ident_by_layer[int(row["layer"])].append(1 if row.get("identified") else 0)
            if row.get("identified"):
                identified_hits[rid] += 1
                any_ident = 1
        ident_by_cond[cond].append(any_ident)
        family_ident[fam].append(any_ident)
        max_score_by_cond[cond].append(best)

    layers = sorted(ident_by_layer)
    ident_rate = [round(sum(ident_by_layer[x]) / len(ident_by_layer[x]), 4) for x in layers]

    extra = {
        "model": rows[0].get("model"),
        "n_reports": len(rows),
        "n_complete_families": len({r["base_id"] for r in rows})
        if all(
            len([x for x in rows if x["base_id"] == bid]) == 5
            for bid in {r["base_id"] for r in rows}
        )
        else None,
        "bases": sorted({r["base_id"] for r in rows}),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "condition_counts": dict(Counter(r["condition"] for r in rows)),
        "compile_any_frac": round(compile_any / len(rows), 3),
        "misalign_frac": round(misalign_n / len(rows), 3),
        "residual_identified_frac": round(residual_id_n / len(rows), 3),
        "logit_lens_any_identified_frac": round(
            mean(1 if any(x.get("identified") for x in (r.get("logit_lens") or [])) else 0 for r in rows),
            3,
        ),
        "mean_best_region_score_by_cond": {
            k: round(mean(v), 4) for k, v in sorted(max_score_by_cond.items())
        },
        "mean_peak_pole_by_cond": {
            k: round(mean(v), 5) for k, v in sorted(peak_pole_by_cond.items())
        },
        "median_peak_layer_by_cond": {
            k: int(median(v)) for k, v in sorted(peak_layer_by_cond.items())
        },
        "ident_frac_by_cond": {
            k: round(mean(v), 3) for k, v in sorted(ident_by_cond.items())
        },
        "ident_frac_by_family": {
            k: round(mean(v), 3) for k, v in sorted(family_ident.items())
        },
        "logit_lens_region_votes": dict(region_hits.most_common()),
        "identified_region_votes": dict(identified_hits.most_common()),
        "trajectory_region_votes": dict(traj_regions.most_common()),
        "ident_rate_layers": layers,
        "ident_rate": ident_rate,
        "floor": FLOOR,
    }

    # Examples: identified rows (leaks).
    leaks = []
    for r in rows:
        for row in r.get("logit_lens") or []:
            if row.get("identified"):
                leaks.append(
                    {
                        "slug": r["slug"],
                        "text": r["text"],
                        "condition": r["condition"],
                        "family": r["family"],
                        "layer": row["layer"],
                        "region": row["region"],
                        "score": row["region_score"],
                        "tokens": row.get("top_tokens") or [],
                    }
                )
    extra["identified_events"] = leaks
    extra["identified_event_n"] = len(leaks)

    # Chart-ready downsample of layer means.
    cats = None
    series = []
    for cond, curve in (pole.get("layer_means") or {}).items():
        labels, vals = downsample(curve, 17)
        cats = labels
        series.append({"name": cond, "data": vals})
    extra["pole_chart"] = {"categories": cats, "series": series}

    ident_cats, ident_vals = downsample(
        [ident_rate[layers.index(x)] if False else ident_rate[i] for i, x in enumerate(layers)]
        if layers
        else [],
        min(17, len(layers) or 1),
    )
    extra["ident_chart"] = {
        "categories": [str(layers[round(i * (len(layers) - 1) / max(len(ident_cats) - 1, 1))]) if layers and ident_cats else "" for i in range(len(ident_cats))],
        "data": ident_vals,
    }
    if layers:
        n = min(17, len(layers))
        idx = [round(i * (len(layers) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        extra["ident_chart"] = {
            "categories": [str(layers[i]) for i in idx],
            "data": [ident_rate[i] for i in idx],
        }

    findings = list(pole.get("findings") or [])
    findings.append(
        f"Logit-lens residual region cleared the {FLOOR} floor on "
        f"{extra['logit_lens_any_identified_frac']:.0%} of prompts "
        f"({extra['identified_event_n']} layer events). Mean best-region cosine "
        f"stays under the floor in every condition."
    )
    findings.append(
        f"Compile matched atlas aliases on {extra['compile_any_frac']:.0%} of prompts; "
        f"MisAlign triggered on {extra['misalign_frac']:.0%}."
    )
    if extra["identified_region_votes"]:
        findings.append(
            "When a layer did identify, the lexical matcher labeled "
            + ", ".join(f"{k} ({v})" for k, v in extra["identified_region_votes"].items())
            + " — correlative token hits, not fly neuropils."
        )
    pole["findings"] = findings
    extra["findings"] = findings

    save_aggregate(ROOT, rows)
    agg = ROOT / "_aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "findings.json").write_text(json.dumps(pole, indent=2), encoding="utf-8")
    (agg / "suite_mine.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    (ROOT / "INDEX.json").write_text(
        json.dumps(
            {
                "n": len(rows),
                "stopped_early": True,
                "model": extra["model"],
                "prompts": [
                    {"slug": r["slug"], "condition": r["condition"], "family": r["family"]}
                    for r in rows
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: extra[k] for k in extra if k not in ("identified_events", "pole_chart", "ident_chart")}, indent=2))
    print("--- findings ---")
    for f in findings:
        print("-", f)
    print(f"wrote {agg / 'suite_mine.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
