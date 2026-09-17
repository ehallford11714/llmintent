# LLMIntent 1,440-case actuation study — 27B findings

Measured on the local Qwen/Qwen3.8-27B NF4 snapshot `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. No GPT-2, unsloth, or synthetic-activation substitute was used. Confirmation was not used to choose sites, C, or patch contrasts.

Provenance check: [`provenance.json`](provenance.json). Plain-language summary: [`PLAIN_ENGLISH.md`](PLAIN_ENGLISH.md).

## Model and kit

| Item | Value |
|---|---|
| Checkpoint | `C:\Users\ehall\.cache\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Class | `Qwen3_5ForConditionalGeneration` |
| Text blocks / hidden | 64 / 5120 |
| Final norm | `Qwen3_5RMSNorm` |
| Precision | bitsandbytes NF4, double quant, bfloat16 compute |
| LLMIntent | source `C:\Users\ehall\Desktop\research\llmintent\src`, git `752512aeeb68fd08d53ca9cc5283f2caf31b9090` (v1.7.1) |
| Dataset SHA-256 | `db709fbb000adc557278de0c64bbf84733101eefb6068504f066a132ab94e5d0` |
| Cases | 1,440 unique IDs; 0 missing, 0 extra; discovery 960 / calibration 240 / confirmation 240 |
| Capture | last encoded prompt token (assistant prefix, token id 25); not last content word |
| Choice tokens A–F | 357, 417, 351, 414, 458, 426 (six distinct space-prefixed singles) |
| Identity error | 0.0 on first-case hook check for every collect split |

The runner refuses GPT-2, unsloth, non-four-bit loads, non-64 text-block adapters, and vision blocks. Linear attention used the slow PyTorch fallback (`causal_conv1d` / `flash-linear-attention` not installed).

Confirmation has three template families and four objects. Do not treat 1,440 cases as independent replications.

## Baseline accuracy (27B NF4)

| Split | n | Overall | Frame track | Policy track |
|---|---:|---:|---:|---:|
| Pilot (engineering) | 24 | 0.917 | — | — |
| Discovery | 960 | 0.945 | 0.908 | 1.000 |
| Calibration | 240 | 0.896 | 0.826 | 1.000 |
| Confirmation | 240 | 0.912 | 0.868 | 0.979 |

Confirmation by status: commit 1.00, quote 1.00, withhold 1.00, scene_only 1.00, conditional 0.75, reflect 0.458. Policy remains high. Calibration quote was the weakest status at 0.625. 99 / 1,440 items incorrect ([`failure_ledger.json`](failure_ledger.json)). Most errors are frame-status, especially quote (discovery/calibration) and reflect (confirmation).

Human review of 12 items: well-formed; possible quote/commit ambiguity on `calibration:t7:disk:toward:frame:commit` noted, not edited.

## Claim 1 — Decodability (availability)

**Supported for action status and STEP/WAIT at late residual sites. Not supported as non-lexical scene/motion decoding.**

Frozen linear probes (OneVsRest + liblinear) were fit on discovery, C selected on calibration, then evaluated on confirmation. 66 sites: `block.0.out`–`block.63.out`, `final_norm.pre`, `final_norm.post`.

Confirmation balanced accuracy (frozen selected site vs lexical vs 5-shuffle mean):

| Target | Selected site (from calibration) | Confirmation bal. acc. | Lexical | Shuffle mean |
|---|---|---:|---:|---:|
| Frame (6-way status) | `block.50.out` | 0.938 | 0.708 | 0.175 |
| Motion (toward/away) | `block.4.out` | 0.867 | 1.000 | 0.516 |
| Action (STEP/WAIT) | `block.33.out` | 0.812 | 0.500 | 0.488 |

Depth pattern on confirmation:

- Frame is near chance early (`block.4` 0.431, `block.31` 0.424) and high late (`block.50` 0.938, `block.60` / `final_norm.post` 0.931). Beats lexical and shuffles at the selected late site.
- Motion is high early (`block.0` 0.925, `block.4` 0.867) and weaker late (~0.70–0.78). Lexical is 1.0 on both calibration and confirmation: early motion decoding is surface words, not a hidden scene code.
- Action is chance-level lexical (0.5). Calibration action is 0.854 at `block.4` and 1.0 from about `block.33` onward. Confirmation action is 0.812 at the selected `block.33` and 0.979 at `block.60` / `final_norm.post`.

Commit-vs-reflect AUC on confirmation: `block.4` 0.773, `block.31` 0.729, `block.50` 0.896, `block.60` 0.912, `final_norm.post` 0.837. AUC is ranking, not causal use.

Last-block vs pre/post-norm: frame and action remain high at `final_norm.pre` and `final_norm.post` (confirmation frame 0.931, action 0.979). There is no collapse of held-out decoding at the explicit final norm on this task.

Probes are association tests. They do not show that the model uses the decoded information.

## Claim 2 — Causal use

**Supported at `block.60.out` for scene, policy, and frame (commit↔reflect). Unsupported at `block.4.out` and `block.31.out`.**

Frozen shortlist (written from discovery/calibration only, before confirmation patches): sites `block.4.out`, `block.31.out`, `block.60.out`; contrasts scene / policy / frame; max_pairs 24; seed 20260916; 5 random-direction controls; identity patch; same-answer donor. Rationale is in [`frozen_patch_selection.json`](frozen_patch_selection.json).

Effect = source-answer minus target-answer log-probability gap. Identity error = 0.0 on all 18 jobs.

Confirmation mean effect (n=24 pairs each):

| Contrast | `block.4.out` | `block.31.out` | `block.60.out` | Random @60 | Same-label @60 | Normalized transfer @60 |
|---|---:|---:|---:|---:|---:|---:|
| Frame | 0.016 | 0.055 | 5.003 | 0.032 | −0.036 | 0.743 |
| Policy | 0.000 | −0.008 | 5.898 | −0.096 | 0.451 | 0.783 |
| Scene | 0.016 | 0.005 | 5.711 | −0.101 | −0.018 | 0.794 |

Calibration exploratory patches (allowed; not confirmation) show the same pattern: ~0 at 4 and 31; large at 60 (frame 5.54, policy 9.20, scene 8.96; normalized transfer ~0.81).

Both-correct subset at confirmation `block.60.out`: frame 13/24 pairs, policy 22/24, scene 23/24. Frame both-correct mean effect is 6.99.

Caveats:

- Policy same-answer donor at confirmation `block.60.out` is 0.45, not ~0. Directional transfer still exceeds that control and random, but the donor is not a clean null.
- No commit↔withhold or commit↔quote confirmation patches were run (quote was not a primary frozen contrast).
- Optional downstream restoration was not run (`rescue_site` null).
- Effects are at one token position (assistant prefix). NF4-only.

## Claim 3 — Selectivity / depth specialization

**Partially supported for decoding; not supported as a causal site×task dissociation on the frozen shortlist.**

Decoding: motion/scene information is available early and is lexical; frame status and STEP/WAIT emerge mid-to-late and beat lexical/shuffles.

Causal: all three frozen contrasts transfer at `block.60.out` and none at `block.4.out` or `block.31.out`. That is a late-only causal profile, not a scene-early vs policy-late causal split. A site×task interaction test was not computed. Scene and policy interventions at 60 are both large; this does not by itself prove functional specialization versus a late residual that carries several decision features.

## Claim 4 — Specific units / subspaces

**Untested.** No FFN-unit or subspace isolation, no matched-count control units, no targeted rescue. Block-level whole-vector patches only.

## Claim 5 — Fly-prior discovery benefit

**Untested.** No fly-inspired candidate selector versus equal-budget control. LLMIntent selector caveats in the kit README were not addressed.

## What this suggests about where actuation is represented

On this 27B residual stream, at the assistant-prefix token:

1. Scene direction is readable early because the words are in the prompt. Patching early motion-rich residuals does not move STEP/WAIT logits.
2. Action status and the STEP/WAIT decision become linearly readable around the mid-late blocks and remain readable through final norm.
3. Interchanging the whole residual at `block.60.out` moves the relevant answer logits for scene-swap, policy-swap, and commit↔reflect. The same interchange at `block.4` or `block.31` does not. Information can be present (or even probe-decodable) without being causally used at that site for the letter choice.
4. This is compatible with “available early, used late,” not with a simple story that intention first appears only at the last layer, and not with a demonstrated neuron circuit or fly homology.

## Isolated harness (not in this package)

The 1,440-case runner lived in a local experiment directory. It was not merged into the `llmintent` import API. Weights were not edited.

- 27B-only guards (refuse GPT-2 / unsloth / non-four-bit / non-64 text blocks / vision blocks)
- `OneVsRestClassifier(LogisticRegression liblinear)` for six-way frame probes
- One-load collect and patch wrappers

Live 27B hooks passed with identity error 0.0. Per-token `.npz` activations (~1.2 GB) are not in this repo.

## Shipped files

- [`PLAIN_ENGLISH.md`](PLAIN_ENGLISH.md), this file
- [`benchmark.jsonl`](benchmark.jsonl) — SHA-256 `db709fbb000adc557278de0c64bbf84733101eefb6068504f066a132ab94e5d0`
- [`provenance.json`](provenance.json), [`run_*.json`](run_confirmation.json), [`baseline_*.json`](baseline_confirmation.json)
- [`frozen_probes.json`](frozen_probes.json), [`probe_confirmation_compact.json`](probe_confirmation_compact.json)
- [`patch_summary.json`](patch_summary.json), [`frozen_patch_selection.json`](frozen_patch_selection.json)
- [`failure_ledger.json`](failure_ledger.json), [`auc_commit_reflect_confirmation.json`](auc_commit_reflect_confirmation.json)
