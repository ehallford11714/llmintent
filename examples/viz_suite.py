"""Visualization suite: morpheme maps, trajectory heatmaps, correlations, animations."""

from llmintent import LLMIntentAnalyzer

PROMPT = "Question: Eight minus two equals ? Answer:"
COT = (
    "Question: Eight minus two equals ? Answer: Let's think step by step. "
    "Eight minus two is six."
)
CONCEPTS = ["subtraction", "eight", "step by step"]


def main() -> None:
    analyzer = LLMIntentAnalyzer(
        "gpt2",
        load_glove=True,
        fit_jspace_transport=False,
    )
    try:
        # Full report: maps + correlation matrices + GIF animations
        paths = analyzer.visualize_report(
            PROMPT,
            twin_b=COT,
            concepts=CONCEPTS,
            output_dir="llmintent_viz",
            include_morphemes=True,
        )
        print("Generated visualizations:")
        for name, path in paths.items():
            print(f"  {name}: {path}")

        # Or use the suite directly for individual plots
        viz = analyzer.visualizer("llmintent_viz")
        mapping = viz.trajectory_mapping(PROMPT, twin_b=COT, concepts=CONCEPTS)
        viz.save_trajectory_map(mapping, filename="trajectory_custom.png")
        viz.save_concept_correlation(mapping, filename="concept_corr_custom.png")
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
