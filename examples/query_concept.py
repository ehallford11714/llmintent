"""Example: query a semantic concept against an activation trajectory."""

from llmintent import LLMIntentAnalyzer


def main() -> None:
    prompt = "Question: A spider has 8 legs. Remove 2. Answer:"
    twin_b = (
        "Question: A spider has 8 legs. Remove 2. How many remain? "
        "Answer: Let's think step by step. Eight minus two equals"
    )

    analyzer = LLMIntentAnalyzer("gpt2", load_glove=False)
    try:
        for concept in ["spider", "eight", "subtraction", "step by step"]:
            result = analyzer.query_concept(concept, prompt, twin_b=twin_b)
            print(f"\n=== Concept: {concept!r} ===")
            print(f"  Peak layer: L{result.peak_layer}")
            print(f"  Matched layers: {result.matched_layers}")
            print(result.knn_ranking[["layer", "knn_similarity", "fused_activation_score", "pivot_tags"]].head(3))

        print("\n=== Full trajectory (concept: subtraction) ===")
        sub = analyzer.query_concept("subtraction", prompt, twin_b=twin_b)
        cols = ["layer", "concept_similarity", "concept_activation", "kl_weight", "is_activation_pivot"]
        print(sub.trajectory[cols])
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
