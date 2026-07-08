"""Example: unified activation trajectory mapping."""

from llmintent import LLMIntentAnalyzer


def main() -> None:
    prompt = "Question: Eight minus two equals ? Answer:"
    twin_b = "Question: Eight minus two equals ? Answer: Let's think step by step. Eight minus two is"

    analyzer = LLMIntentAnalyzer("gpt2", load_glove=False)
    try:
        mapping = analyzer.trajectory_map(
            prompt,
            twin_b=twin_b,
            concepts=["subtraction", "eight", "step by step"],
        )
        print(f"Model: {mapping.model_name}  Layers: {mapping.num_layers}")
        print(f"Pivots: {mapping.pivots}")

        cols = [
            "layer", "regime", "top_intent", "entropy", "kl_divergence",
            "dominant_module", "is_activation_pivot",
        ]
        print(mapping.layers[cols].to_string(index=False))

        for concept, hit in mapping.concept_hits.items():
            print(f"\nConcept {concept!r} → peak L{hit.peak_layer}, matched {hit.matched_layers}")
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
