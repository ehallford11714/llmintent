"""Run complete HellaSwag validation benchmark with progress logging."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmintent.benchmark import BenchmarkRunConfig, HellaSwagBenchmarkRunner, parse_conditions
from llmintent.benchmark.hellaswag import load_hellaswag_jsonl


def main() -> None:
    store_path = ROOT / "llmintent_retraces" / "hellaswag_full.jsonl"
    log_path = ROOT / "llmintent_retraces" / "hellaswag_full.log"
    store_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_hellaswag_jsonl()
    print(f"Loaded {len(examples)} HellaSwag validation examples", flush=True)

    config = BenchmarkRunConfig(
        models=["gpt2", "distilgpt2"],
        conditions=parse_conditions("default"),
        limit=len(examples),
        store_path=str(store_path),
        measure_focus=False,
        use_fallback=False,
    )
    runner = HellaSwagBenchmarkRunner(config)
    started = time.time()

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"examples={len(examples)} models={config.models} conditions={[c.value for c in config.conditions]}\n")
        log.flush()

        results = runner.run_all()
        elapsed = time.time() - started
        compare = runner.compare_from_store()

        payload = {
            "elapsed_sec": elapsed,
            "run_results": results.to_dict(orient="records"),
            "store_summary": compare.to_dict(orient="records") if not compare.empty else [],
            "store_path": str(store_path),
        }
        out_path = ROOT / "llmintent_retraces" / "hellaswag_full_results.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        log.write(f"finished={time.strftime('%Y-%m-%d %H:%M:%S')} elapsed_sec={elapsed:.1f}\n")
        log.write(json.dumps(payload, indent=2))
        log.flush()

    print(json.dumps(payload, indent=2), flush=True)
    compare.to_csv(ROOT / "llmintent_retraces" / "hellaswag_full_results.csv", index=False)
    print(f"Saved {store_path}", flush=True)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
