# What we found (Qwen 3.8-27B)

The fly connectome is a **prior for operations** (sense vs act, withhold vs commit). It is not a claim that the model performs a fly function.

Target notions:

1. **Intention of actuation** — commit to an act that has not happened
2. **Reflection of actuation** — represent an act already taken
3. **Withhold actuation** — choose not to act
4. **Sensory motion / audio** — scene without an act frame
5. **Quoted actuation** — reported speech about an act (control)

Model: `Qwen/Qwen3.8-27B` (NF4). Never GPT-2. Temporary inference only; no weight edits.

Raw traces: [`artifacts/function_mine_27b/`](artifacts/function_mine_27b/).

## 100-prompt validation

Late residual cosine does **not** dissociate intention from reflection. Next-token logits **do**.

| Condition | n | Residual intend | Residual reflect | Δ residual | Logit intend−reflect | Action−wait | Winning late probe |
|-----------|---|-----------------|------------------|------------|----------------------|-------------|--------------------|
| Intention of actuation | 20 | 0.311 | 0.267 | +0.044 | **+8.27** | +1.00 | intention |
| Reflection of actuation | 20 | 0.263 | 0.227 | +0.036 | **−6.23** | −2.10 | intention |
| Withhold actuation | 15 | 0.358 | 0.312 | +0.046 | +3.46 | **−4.19** | intention |
| Quoted actuation | 10 | 0.324 | 0.197 | +0.127 | **−2.48** | +1.21 | intention |
| Sensory motion | 15 | 0.807 | 0.822 | −0.016 | +0.40 | −0.25 | sensory motion |
| Sensory audio | 10 | 0.851 | 0.874 | −0.023 | −1.29 | −2.01 | sensory audio |
| Static scene | 10 | 0.727 | 0.693 | +0.034 | +0.70 | −0.17 | intention |

`dissociated` on residual cosine: **false**. Intention prompts prefer the intention residual only by a thin margin; reflection prompts still prefer the intention residual.

Quoted speech is the split: the residual looks most like intention (Δ +0.127) while logits prefer past (−2.48). Sensory scenes sit at ~0.8 cosine to every probe — global residual similarity, not a specific notion.

## Where intention of actuation emerges

Scored all **65** layers (0 = embed, 64 = last residual) on the 20 “I am going to…” prompts, same-layer cosine vs notion probes.

| Layer | Intend cos | Reflect cos | Motion cos | Δ intend−reflect | Share top intend | Winning probe |
|-------|------------|-------------|------------|------------------|------------------|---------------|
| 0 embed | 0.242 | 0.242 | 0.242 | 0.000 | 1.00* | tie / first key |
| 10 | 0.916 | 0.910 | 0.903 | +0.006 | 0.50 | intention |
| 16 early | 0.874 | 0.842 | 0.866 | +0.031 | 0.65 | intention |
| 18 | 0.830 | 0.797 | 0.838 | +0.033 | 0.20 | sensory motion |
| 32 mid | 0.642 | 0.660 | 0.700 | −0.018 | 0.05 | sensory motion |
| 48 | 0.600 | 0.622 | 0.627 | −0.021 | 0.00 | sensory motion |
| 56 | 0.483 | 0.470 | 0.457 | +0.013 | 0.85 | intention |
| 61 | 0.547 | 0.536 | 0.535 | +0.011 | 0.75 | intention |
| 64 last | 0.311 | 0.267 | 0.211 | +0.044 | 1.00 | intention |

\*Layer 0 is a tie. Do not read that as emergence.

**Late regime starts at layer 61.** It then holds through 64 (20/20). Middle layers 18–52 are occupied by the scene.

Full curve: [`artifacts/function_mine_27b/layer_emergence.json`](artifacts/function_mine_27b/layer_emergence.json).

## What this suggests

Intention of actuation is a **late readout**, not a mid-depth occupant. The stack carries the scene; the last layers tilt that scene toward “will” rather than “did.” Sense-then-act is a useful order hint from the fly prior and a bad literal map onto 27B anatomy.

This is observational (probe cosine + next-token logprobs). It is not a causal circuit claim.

## Reproduce

```powershell
python -m llmintent function-mine --model qwen:27b --out examples/artifacts/function_mine_27b
```

27B-only. GPT-2 is rejected.
