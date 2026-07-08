"""Example: J-space layer thoughts and activation layer identification."""

from llmintent import LLMIntentAnalyzer


def main() -> None:
    prompt = "Question: A spider has 8 legs. If you remove 2, how many remain? Answer:"
    analyzer = LLMIntentAnalyzer("gpt2", fit_jspace_transport=True, load_glove=False)

    try:
        report = analyzer.analyze_prompt(prompt, track_tokens=["8", "6", "spider"])
        print("=== Activation Layers ===")
        for name, layer in report.activation_layers.items():
            print(f"  {name}: layer {layer}")

        print("\n=== Layer Correspondence (sample) ===")
        print(report.layer_map[["layer", "regime", "role", "top_intent", "is_activation_pivot"]].head(8))

        print("\n=== Workspace Band Summary ===")
        summary = analyzer.layer_band_summary(prompt)
        print(f"  Bands: {summary['regime_bands']}")
        print(f"  Workspace thoughts: {summary['workspace_thoughts']}")

        if report.intent_trace:
            print("\n=== Layer Thoughts at final token ===")
            for layer in [0, 3, 6, 11, 12]:
                if layer < report.intent_trace.num_layers:
                    thought = report.intent_trace.top_thought_at(layer)
                    print(f"  L{layer}: {thought!r}")
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
