"""Example: cognitive module kernels via KL + twin Barlow minimization."""

from llmintent import LLMIntentAnalyzer


def main() -> None:
    twin_a = "I have five apples and I eat two. I now have exactly"
    twin_b = (
        "Question: I have five apples and eat two. How many remain? "
        "Answer: Let's think step by step. Five minus two equals"
    )

    analyzer = LLMIntentAnalyzer("gpt2", load_glove=False)
    try:
        profile = analyzer.cognitive_modules(twin_a, twin_b)
        print("=== Cognitive Kernels (KL + Twin Barlow) ===")
        for k in profile.kernels:
            print(
                f"  {k.module:15} L{k.layer:2d}  score={k.score:.3f}  "
                f"KL={k.kl_weight:.3f}  intent={k.top_intent!r}"
            )

        print("\n=== Layer Assignments (sample) ===")
        cols = ["layer", "dominant_module", "kl_divergence", "reasoning", "meta_reasoning"]
        print(profile.layer_assignments[cols].head(8))

        print("\n=== Module Layer Bands ===")
        for module, layers in profile.to_dict()["module_layers"].items():
            print(f"  {module}: {layers}")
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
