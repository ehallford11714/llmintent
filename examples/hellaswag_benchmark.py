"""
HellaSwag benchmark: focused vs extreme retrace ablations on SLMs.

Requires: pip install llmintent[benchmark]
Uses fallback fixtures if datasets unavailable (--fallback).
"""

from llmintent.benchmark import (
    BenchmarkRunConfig,
    HellaSwagBenchmarkRunner,
    RetraceStore,
    list_slms,
    parse_conditions,
)

# Prepared SLMs for comparison
MODELS = ["gpt2", "distilgpt2"]
STORE_PATH = "llmintent_retraces/hellaswag.jsonl"


def main() -> None:
    print("Prepared SLMs:")
    for slm in list_slms():
        print(f"  {slm['key']:12} {slm['hf_name']:25} ({slm['params_m']}M params)")

    config = BenchmarkRunConfig(
        models=MODELS,
        conditions=parse_conditions("fast"),
        limit=10,
        store_path=STORE_PATH,
        use_fallback=True,  # set False when datasets installed
        measure_focus=True,
    )

    runner = HellaSwagBenchmarkRunner(config)
    print(f"\nRunning HellaSwag ablations (limit={config.limit})...")
    results = runner.run_all()
    print("\nRun results:")
    print(results.to_string(index=False))

    store = RetraceStore(STORE_PATH)
    summary = store.summarize_accuracy()
    print("\nStored retracement summary:")
    print(summary.to_string(index=False))

    csv_path = store.export_csv("llmintent_retraces/hellaswag_results.csv")
    print(f"\nExported: {csv_path}")
    print(f"Retracements JSONL: {STORE_PATH}")


if __name__ == "__main__":
    main()
