"""Basic usage example mirroring notebook steering analysis."""

from llmintent import LLMIntentAnalyzer


def main() -> None:
    analyzer = LLMIntentAnalyzer("distilbert-base-uncased", load_glove=False)
    prompt = "The quick brown fox jumps over the lazy"
    report = analyzer.analyze_prompt(prompt)
    print("Intensity sweep (first 5 layers):")
    print(report.intensity_sweep.head())
    print("\nEntropy trajectory (first 5 layers):")
    print(report.entropy_trajectory.head())
    analyzer.cleanup()


if __name__ == "__main__":
    main()
