# Actuation notions (27B)

Does the residual separate intention of actuation from reflection of actuation, and from withhold and sensory-only scenes?

The fly brain is a prior for sense-versus-act structure. The model is not expected to perform a fly function. We mine intention of actuation, reflection of actuation, and related notions.

Model `Qwen/Qwen3.8-27B` · 100 prompts

## Findings

{
  "intention_prompts_prefer_intention_residual": true,
  "reflection_prompts_prefer_reflection_residual": false,
  "intention_margin": 0.044,
  "reflection_margin": 0.0363,
  "dissociated": false
}

## Condition means (late residual)

- **intention_of_actuation** n=20: intention +0.311, reflection +0.267, Δ +0.044, top `intention_of_actuation`
- **quoted_actuation** n=10: intention +0.324, reflection +0.197, Δ +0.127, top `intention_of_actuation`
- **reflection_of_actuation** n=20: intention +0.263, reflection +0.227, Δ +0.036, top `intention_of_actuation`
- **sensory_audio** n=10: intention +0.851, reflection +0.874, Δ -0.023, top `sensory_audio`
- **sensory_motion** n=15: intention +0.807, reflection +0.822, Δ -0.016, top `sensory_motion`
- **static_scene** n=10: intention +0.727, reflection +0.693, Δ +0.034, top `intention_of_actuation`
- **withhold_actuation** n=15: intention +0.358, reflection +0.312, Δ +0.046, top `intention_of_actuation`