"""LLMIntent Live — real-time demo with a loaded SLM."""

from llmintent.live import LiveIntentPipeline, LiveSessionConfig, list_live_models


def main() -> None:
    print("Registered live models:")
    for m in list_live_models():
        print(f"  {m['key']:14} {m['hf_name']}")

    model = "gpt2"  # swap to qwen-0.5b or phi3-mini when GPU/RAM allows
    pipe = LiveIntentPipeline(LiveSessionConfig(model_key=model, retracement_mode="focus_gate"))

    prompt = "Question: If a train leaves at 2pm and travels 60 miles in 2 hours, what is its speed? Answer:"

    try:
        print(f"\nLoading {model}…")
        pipe.load()

        print("\n--- Analyze ---")
        analysis = pipe.analyze(prompt)
        print(analysis.to_dict())

        print("\n--- Heighten ---")
        heighten = pipe.heighten(prompt, steer=False)
        print(f"focus {heighten.focus_before:.3f} → {heighten.focus_after:.3f}")

        print("\n--- Probe (focus_gate) ---")
        for tok, prob in pipe.probe_next_tokens(prompt, k=5):
            print(f"  {tok!r}: {prob:.3f}")

        print("\n--- Generate ---")
        gen = pipe.generate(prompt, max_new_tokens=32, retracement_mode="focus_gate")
        print(gen.completion)
    finally:
        pipe.unload()


if __name__ == "__main__":
    main()
