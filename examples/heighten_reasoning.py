"""Heightened reasoning: diagnose focus, force retrace, measure gain."""

from llmintent import LLMIntentAnalyzer

PROMPT = "Question: Eight minus two equals ? Answer:"
COT = (
    "Question: Eight minus two equals ? Answer: Let's think step by step. "
    "We need to subtract two from eight. Eight minus two equals six."
)
CONCEPTS = ["subtraction", "eight", "six"]


def main() -> None:
    analyzer = LLMIntentAnalyzer("gpt2", load_glove=False)
    try:
        # 1. Diagnose whether reasoning is focused or diffuse
        focus, mapping = analyzer.diagnose_focus(
            PROMPT,
            anchor_prompt=COT,
            concepts=CONCEPTS,
        )
        print("Baseline focus:")
        print(f"  focus_score={focus.focus_score:.3f}  needs_retrace={focus.needs_retrace}")
        print(f"  reasoning_concentration={focus.reasoning_concentration:.3f}")
        print(f"  recommended_layers={focus.recommended_focus_layers}")

        # 2. Heighten via forced self-retrace scaffold
        result = analyzer.heighten_reasoning(
            PROMPT,
            anchor_prompt=COT,
            concepts=CONCEPTS,
            mode="explicit_retrace",
            apply_steering=False,
        )

        print("\nRetrace plan:")
        print(f"  mode={result.plan.mode.value}")
        print(f"  retrace_layers={result.plan.retrace_layers}")
        print(f"\nRetrace prompt preview:\n{result.plan.retrace_prompt[:200]}...")

        print("\nFocus gain after retrace:")
        for k, v in result.focus_gain.items():
            print(f"  {k}: {v:+.4f}")

        print(f"\nHeightening successful: {result.heightening_successful}")

        # 3. Optional: activation steering at reasoning layers
        if result.baseline_focus.needs_retrace:
            steered = analyzer.heighten_reasoning(
                PROMPT,
                anchor_prompt=COT,
                concepts=CONCEPTS,
                mode="concept_anchor",
                apply_steering=True,
            )
            print("\nSteering report:")
            print(steered.steering_report)
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
