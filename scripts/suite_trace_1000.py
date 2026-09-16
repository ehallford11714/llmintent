"""1000-prompt full-suite trace: densities, logit-lens regions, charts, per-prompt dirs.

200 bases x 5 ablation conditions = 1000 texts. One 27B (or other) forward
per text. Each prompt writes its own folder. Aggregate charts at the end.

Does not claim a number circuit or fly neuropils. Residual region labels
are identified only if cosine >= 0.18.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pole_ablation_100 import Item, build_corpus, conditions_for, mine

FLOOR = 0.18
CONDS = (
    "original",
    "with_number_5",
    "changed_number_10",
    "drop_noun",
    "swap_noun",
)


def build_200_bases() -> list[Item]:
    base = build_corpus()
    extra: list[Item] = []
    for it in base:
        extra.append(
            Item(
                id=it.id + 100,
                family=it.family,
                original=f"Yesterday I noticed: {it.original}",
                noun=it.noun,
                has_number=it.has_number,
            )
        )
    return base + extra


def iter_1000() -> list[dict]:
    out: list[dict] = []
    n = 0
    for item in build_200_bases():
        for cond, text in conditions_for(item).items():
            out.append(
                {
                    "index": n,
                    "base_id": item.id,
                    "family": item.family,
                    "condition": cond,
                    "text": text,
                    "noun": item.noun,
                }
            )
            n += 1
    if len(out) != 1000:
        raise RuntimeError(f"expected 1000 prompts, got {len(out)}")
    return out


def _slug(row: dict) -> str:
    return f"p{row['index']:04d}_{row['condition']}"


def _save_charts(folder: Path, payload: dict) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    charts = folder / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    pole = payload.get("pole_intensity") or []
    if pole:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(range(len(pole)), pole, color="#1f4e79", lw=1.6)
        ax.set_title("Numerical pole intensity by layer (last token)")
        ax.set_xlabel("Layer index (0 = embeddings)")
        ax.set_ylabel("Cosine intensity")
        ax.set_ylim(0, max(0.12, max(pole) * 1.15))
        fig.tight_layout()
        p = charts / "pole_intensity.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))

    spans = payload.get("span_series") or {}
    if spans:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        x = list(range(len(next(iter(spans.values())))))
        for rid, series in spans.items():
            if max(series) <= 0:
                continue
            ax.plot(x, series, marker="o", label=rid)
        ax.set_title("Region occupancy through prompt spans")
        ax.set_xlabel("Span index")
        ax.set_ylabel("Compile occupancy")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        p = charts / "region_span_density.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))

    lens = payload.get("logit_lens") or []
    if lens:
        from llmintent.anatomy.atlas import region_ids

        rids = list(region_ids())
        mat = np.zeros((len(rids), len(lens)))
        for j, row in enumerate(lens):
            rid = row.get("region") or "workspace"
            sc = float(row.get("region_score") or 0)
            if rid in rids:
                mat[rids.index(rid), j] = sc
        fig, ax = plt.subplots(figsize=(10, 4.2))
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=max(0.3, float(mat.max())))
        ax.set_yticks(range(len(rids)))
        ax.set_yticklabels(rids, fontsize=8)
        ax.set_xlabel("Sampled layer (logit lens)")
        ax.set_title("Logit-lens region score map (identify only if >= 0.18)")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="region cosine")
        fig.tight_layout()
        p = charts / "logit_lens_region_map.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(
            [r["layer"] for r in lens],
            [r["region_score"] for r in lens],
            color="#8c2d19",
            marker=".",
        )
        ax.axhline(FLOOR, ls="--", color="#666", lw=1, label="identify floor 0.18")
        ax.set_title("Logit-lens best-region cosine by layer")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Best region cosine")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = charts / "logit_lens_score.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))

    intents = payload.get("layer_intents_active") or []
    if intents:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(range(len(intents)), intents, color="#2a6f4e")
        ax.set_title("Active catalogue intents per layer (count > 0)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Active intent count")
        fig.tight_layout()
        p = charts / "intent_density.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))

    return paths


def _report_md(payload: dict) -> str:
    lines = [
        f"# {payload.get('slug')}",
        "",
        f"**Model:** `{payload.get('model')}`  ",
        f"**Condition:** `{payload.get('condition')}` · **family:** `{payload.get('family')}`  ",
        f"**Prompt:** {payload.get('text')}",
        "",
        "## Compile densities",
        "",
        f"Regions: {', '.join(payload.get('compile_regions') or ['(none)'])}",
        "",
        "## Logit lens on regions",
        "",
        "| Layer | Band | Tokens | Region | Score | Identified |",
        "|------:|------|--------|--------|------:|:----------:|",
    ]
    for row in payload.get("logit_lens") or []:
        toks = ", ".join(row.get("top_tokens") or [])[:80]
        ident = "yes" if row.get("identified") else "no"
        lines.append(
            f"| {row['layer']} | {row.get('band','')} | {toks} | "
            f"`{row.get('region')}` | {row.get('region_score', 0):.3f} | {ident} |"
        )
    lines += [
        "",
        "## MisAlign Flag",
        "",
        f"{'TRIGGERED' if payload.get('misalign') else 'off'}",
        "",
        "## Charts",
        "",
    ]
    for p in payload.get("charts") or []:
        name = Path(p).name
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def trace_one(bundle, pole, row: dict, *, stride: int, top_k: int) -> dict:
    from llmintent.anatomy.atlas import region_ids
    from llmintent.anatomy.compile import compile_regions
    from llmintent.anatomy.misalign import scan_negative_intent
    from llmintent.anatomy.svd_map import match_text_to_region, svd_hidden_matrix
    from llmintent.anatomy.thoughts import LayerThought, _band_for_depth, _decode_hidden, _sample_layers
    from llmintent.anatomy.trace import trace_prompt
    from llmintent.anatomy.trajectory import trajectory
    from llmintent.forward import forward_hidden_states
    from llmintent.metrics import cosine_intensity

    text = row["text"]
    _, states = forward_hidden_states(bundle, text)
    h0 = states[0][0, -1, :]
    pole_d = pole.to(device=h0.device, dtype=h0.dtype)
    pole_curve = [
        cosine_intensity(states[i][0, -1, :], pole_d) for i in range(len(states))
    ]

    n_blocks = max(len(states) - 1, 1)
    sample = _sample_layers(n_blocks, stride)
    thoughts: list[LayerThought] = []
    lens_rows: list[dict] = []
    last_rows: list[np.ndarray] = []
    for li in sample:
        idx = min(li + 1, len(states) - 1)
        hidden = states[idx][0, -1, :]
        tokens = _decode_hidden(bundle, hidden, top_k)
        blob = " ".join(tokens)
        rid, score = match_text_to_region(blob) if blob else ("workspace", 0.0)
        identified = float(score) >= FLOOR
        depth = li / max(n_blocks - 1, 1)
        band = _band_for_depth(depth)
        thoughts.append(
            LayerThought(
                layer=li,
                depth=depth,
                band=band,
                top_tokens=tokens,
                region=rid if identified else "workspace",
                region_score=float(score),
                residual_l2=float(hidden.float().norm().cpu()),
            )
        )
        lens_rows.append(
            {
                "layer": li,
                "band": band,
                "top_tokens": tokens,
                "region": rid,
                "region_score": round(float(score), 4),
                "identified": identified,
            }
        )
        last_rows.append(hidden.detach().float().cpu().numpy().reshape(-1))

    plan = compile_regions(text)
    trace = trace_prompt(text)
    span_series = {r.id: r.series for r in trace.regions if max(r.series) > 0}
    traj = trajectory(text, thoughts=thoughts, print_flag=False, n_layers=n_blocks)
    _, flag = scan_negative_intent(text, thoughts=thoughts, path=traj.path)

    svd_occ = {}
    if len(last_rows) >= 3:
        H = np.stack(last_rows, axis=0)
        _, S, Vh = svd_hidden_matrix(H, top_k=min(6, H.shape[0]))
        svd_occ = {"singular_values": [round(float(s), 4) for s in S[:6]]}

    active = [len(li.active) for li in traj.layers]
    return {
        "index": row["index"],
        "slug": _slug(row),
        "base_id": row["base_id"],
        "family": row["family"],
        "condition": row["condition"],
        "text": text,
        "model": getattr(bundle, "name", ""),
        "n_states": len(states),
        "compile_regions": plan.regions,
        "compile_occupancy": {k: round(v, 4) for k, v in plan.occupancy().items() if v > 0},
        "dropped": plan.dropped,
        "span_series": span_series,
        "spans": [s.to_dict() for s in trace.spans],
        "pole_intensity": [round(x, 6) for x in pole_curve],
        "pole_peak_layer": int(max(range(len(pole_curve)), key=lambda i: pole_curve[i])),
        "logit_lens": lens_rows,
        "residual_identified": bool(traj.residual_identified),
        "trajectory_path": traj.path,
        "imputed": traj.imputed,
        "layer_intents_active": active,
        "svd": svd_occ,
        "misalign": bool(flag.triggered),
        "misalign_trigger": flag.trigger,
        "atlas_ids": list(region_ids()),
    }


def write_prompt_dir(root: Path, payload: dict) -> Path:
    folder = root / payload["slug"]
    folder.mkdir(parents=True, exist_ok=True)
    charts = _save_charts(folder, payload)
    payload = dict(payload)
    payload["charts"] = [Path(p).name for p in charts]
    (folder / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (folder / "report.md").write_text(_report_md(payload), encoding="utf-8")
    dens = folder / "densities.csv"
    with dens.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "key", "index", "value"])
        for i, v in enumerate(payload.get("pole_intensity") or []):
            w.writerow(["pole", "intensity", i, v])
        for rid, series in (payload.get("span_series") or {}).items():
            for i, v in enumerate(series):
                w.writerow(["span", rid, i, v])
        for row in payload.get("logit_lens") or []:
            w.writerow(["logit_lens", row.get("region"), row.get("layer"), row.get("region_score")])
        for i, v in enumerate(payload.get("layer_intents_active") or []):
            w.writerow(["intents_active", "count", i, v])
    return folder


def save_aggregate(root: Path, rows: list[dict]) -> None:
    agg = root / "_aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    # Reuse mine() on pole curves shaped like the 100-run rows.
    mine_rows = []
    for r in rows:
        mine_rows.append(
            {
                "id": r["base_id"],
                "family": r["family"],
                "condition": r["condition"],
                "has_number": any(ch.isdigit() for ch in r["text"]),
                "curve": r.get("pole_intensity") or [0.0],
                "compile": r.get("compile_occupancy") or {},
            }
        )
    n_layers = max(len(r.get("pole_intensity") or []) for r in rows) if rows else 1
    padded = []
    for r in mine_rows:
        c = list(r["curve"])
        if len(c) < n_layers:
            c = c + [0.0] * (n_layers - len(c))
        r = dict(r)
        r["curve"] = c[:n_layers]
        padded.append(r)
    summary = mine(padded, n_layers)
    (agg / "findings.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    means = summary.get("layer_means") or {}
    if means:
        fig, ax = plt.subplots(figsize=(9, 4))
        for cond, curve in means.items():
            ax.plot(range(len(curve)), curve, label=cond, lw=1.5)
        ax.set_title("Mean numerical pole intensity by ablation (1000 prompts)")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Mean cosine intensity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(agg / "mean_pole_by_condition.png", dpi=130)
        plt.close(fig)

    # Logit-lens identified rate by layer (pooled).
    by_layer: dict[int, list[int]] = {}
    for r in rows:
        for row in r.get("logit_lens") or []:
            by_layer.setdefault(int(row["layer"]), []).append(1 if row.get("identified") else 0)
    if by_layer:
        xs = sorted(by_layer)
        ys = [sum(by_layer[x]) / len(by_layer[x]) for x in xs]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(xs, ys, color="#8c2d19")
        ax.set_title("Fraction of prompts with identified residual region (cosine >= 0.18)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Identified fraction")
        fig.tight_layout()
        fig.savefig(agg / "logit_lens_identified_rate.png", dpi=130)
        plt.close(fig)


def run(
    *,
    model: str,
    out_dir: Path,
    stride: int,
    limit: int | None,
    load_in_4bit: bool | None,
) -> None:
    from llmintent.poles import build_numerical_pole
    from llmintent.suite import load_suite_model, resolve_model_spec

    prompts = iter_1000()
    if limit:
        prompts = prompts[: int(limit)]
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_model_spec(model=model, use_env=False)
    four = load_in_4bit
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    print(
        f"loading {model} 4bit={bool(four)} prompts={len(prompts)} stride={stride} -> {out_dir}",
        file=sys.stderr,
        flush=True,
    )
    bundle = load_suite_model(model=model, load_in_4bit=four)
    try:
        pole = build_numerical_pole(bundle)
    except ValueError:
        pole = build_numerical_pole(bundle, tokens=["0", "1", "2", "3", "4", "5", " 0", " 1"])

    done_rows: list[dict] = []
    for i, row in enumerate(prompts):
        folder = out_dir / _slug(row)
        report = folder / "report.json"
        if report.exists():
            payload = json.loads(report.read_text(encoding="utf-8"))
            done_rows.append(payload)
            print(f"skip {i+1}/{len(prompts)} {row['slug'] if False else _slug(row)}", file=sys.stderr, flush=True)
            continue
        payload = trace_one(bundle, pole, row, stride=stride, top_k=6)
        write_prompt_dir(out_dir, payload)
        done_rows.append(payload)
        print(f"wrote {i+1}/{len(prompts)} {_slug(row)}", file=sys.stderr, flush=True)
        if (i + 1) % 25 == 0:
            save_aggregate(out_dir, done_rows)
            (out_dir / "INDEX.json").write_text(
                json.dumps(
                    {
                        "n": len(done_rows),
                        "model": getattr(bundle, "name", model),
                        "prompts": [
                            {"slug": r["slug"], "condition": r["condition"], "family": r["family"]}
                            for r in done_rows
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    save_aggregate(out_dir, done_rows)
    (out_dir / "INDEX.json").write_text(
        json.dumps({"n": len(done_rows), "model": getattr(bundle, "name", model)}, indent=2),
        encoding="utf-8",
    )
    print(f"done {len(done_rows)} reports in {out_dir}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen:27b")
    p.add_argument("--4bit", dest="load_in_4bit", action="store_true")
    p.add_argument("--stride", type=int, default=4, help="Logit-lens layer stride (1 = every layer)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-o", "--output", default="artifacts/suite_27b_1000")
    args = p.parse_args(argv)
    run(
        model=args.model,
        out_dir=Path(args.output),
        stride=max(int(args.stride), 1),
        limit=args.limit,
        load_in_4bit=True if args.load_in_4bit else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
