"""Retracement Transformer: architecture ablation on perplexity."""

from llmintent.retracement import run_retracement_ablation

MODELS = ["gpt2", "distilgpt2"]


def main() -> None:
    print("Retracement Transformer — perplexity ablation")
    print("Modes: baseline, focus_gate, dual_pass, extreme\n")

    df = run_retracement_ablation(
        models=MODELS,
        fast=True,
        text_limit=16,
    )

    print(df.to_string(index=False))
    print("\nInterpretation:")
    print("  delta_ppl_vs_baseline < 0  → retracement improved perplexity")
    print("  delta_ppl_vs_baseline > 0  → retracement hurt (over-constrained)")


if __name__ == "__main__":
    main()
