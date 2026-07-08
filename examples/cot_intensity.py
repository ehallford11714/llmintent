"""CoT vs direct intensity comparison (GPT-2 style prompts from notebook)."""

from llmintent import LLMIntentAnalyzer


def main() -> None:
    direct = "If I have ten apples and lose three, I have"
    cot = (
        "Question: If I have ten apples and lose three, how many do I have? "
        "Answer: Let's think step by step. First, I start with ten. "
        "Then I subtract three. Ten minus three is"
    )

    analyzer = LLMIntentAnalyzer("gpt2")
    try:
        comparison = analyzer.compare_prompts({"Direct": direct, "CoT": cot})
        pivot = analyzer.analyze_prompt(direct, cot_prompt=cot)

        print("Layer 18-style comparison:", pivot.cot_comparison)
        print("Pivot entropy:", pivot.pivot_entropy)
        print("\nCumulative intensity:")
        for label in ("Direct", "CoT"):
            total = comparison[comparison["prompt_type"] == label]["intensity"].sum()
            print(f"  {label}: {total:.4f}")
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
