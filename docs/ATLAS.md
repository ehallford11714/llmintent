# Atlas v1.6 — experimentally grounded correspondence

## What this is

A function-first map: test a fly circuit, then use weight SVD → signed logits → FFN unit localization to nominate LLM counterparts, then score those candidates on held-out tasks and interventions.

A catalogue of eleven fly-inspired labels is a **hypothesis set**, not the deliverable.

## Evidence modes (fly)

| Mode | Meaning |
|------|---------|
| `recording_reanalysis` | Published recordings re-scored here |
| `published_simulation_reproduction` | Replay of a published simulator |
| `proposed_simulation` | New dynamics on a cited wiring prior |

The first circuit (`lplc2_dnp01_looming_escape`) is a **proposed_simulation**. It is not a living-fly experiment. The workspace `fly-brain` package has a literature-core type graph and a pixel looming proxy; it does **not** contain Lappalainen et al. 2024. MaleCNS/neuPrint was **not retrieved** (`NEUPRINT_SERVER` unset).

Published direction reproduced on held-out loom speeds: expanding > receding on DNp01; LPLC2 silence drops DNp01 more than a T4 control; rescue restores.

## SVD convention

`W_down` is `[d, m]` (residual × FFN units). `W_down = U S Vᵀ`. Component `k` writes `a_k(x) u_k` with `a_k = s_k v_kᵀ z(x)`. An SVD vector is a combination of units, not one neuron. Both signs are scored. Token projections are candidate generators, not labels.

## Latent backend

`llmintent.latent` prefers installed `latentintent` or `latentintentinspect`, else `llmintent.latent_vendor`. No extra similarly named dependency.

## Analogous-region test

Comparison is in a **shared operation space** (approach / recede / static / cue / context / perturbation), not cosine between fly synapses and SVD coordinates.

- Fly vector: DNp01 rates from the looming assay.
- LLM vector: forced-choice logprob margins on matched language items (or unidentified if no checkpoint).
- Alignment: Pearson over ≥3 shared axes, otherwise same-sign (approach − recede) selectivity.
- Content selectivity (cue-word items) is scored separately from computational contribution (quote/negation vs functional paraphrases).

`validate_analogous_region_test` checks the comparison machinery with labeled **synthetic** scorers (matched > shuffled and anti). That pass is not pretrained-model evidence.

## MaleCNS + neurotransmitters + Qwen 27B

Local `fly-brain/data` traced tables (166,700 neurons, 25.6M edges) supply signed synapse counts and `consensus_nt`. First circuits:

- LPLC2 (185, acetylcholine) → DNp01 (2, acetylcholine)
- PAM (316, dopamine) and PPL1 (16, dopamine) onto KC; APL (2, GABA) inhibits KC

Dopamine is **not** treated as extra acetylcholine. The analogue is gain on cue→outcome. Test with `qwen:27b` NF4.

```bash
python -m llmintent atlas --no-model --out artifacts/atlas_v16
python -m llmintent atlas --model gpt2 --out artifacts/atlas_v16
python -m pytest tests/test_atlas.py tests/test_anatomy.py -q
```

## Migration

- `match_text_to_region` abstains below 0.18 (`unidentified`).
- `region_axis_from_anatomy` returns `(None, "unidentified")` unless `allow_random_control=True`.
- `LayerResponsibility.responsible_for` now says **associated with** (unvalidated).
- Giant-fibre short-pipe on “I see a car. Stop.” is an experimental signal, not MisAlign Flag.
- Occupancy-jitter IV remains a **simulation fixture**.
