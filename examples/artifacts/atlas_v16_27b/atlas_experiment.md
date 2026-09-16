# LLMIntent atlas correspondence (v1.6)

Schema `llmintent.anatomy.evidence/v1` · 2026-09-15T18:00:09.667475+00:00

## Fly assay

- Circuit: `malecns_lplc2_dnp01_looming_escape`
- Mode: `proposed_simulation` (living fly: False)
- Validation: malecns_weights_reproduce_published_direction_on_held_out_speeds
- Contributors: LPLC2, DNp01

## Correspondence table

- Fly `['LPLC2', 'DNp01']` → LLM `['model.language_model.layers.0.mlp.down_proj.weight:unit0']`
  - strength: `candidate_association` · fly evidence: proposed_simulation / malecns_weights_reproduce_published_direction_on_held_out_speeds
  - SVD: {'region_id': 'svd:L0:k0:s+1', 'contrast': 0.0, 'sign': 1, 'plus_tokens': [], 'minus_tokens': []}
  - intervention: unidentified
- Fly `['PAM', 'PPL1']` → LLM `[]`
  - strength: `candidate_association` · fly evidence: proposed_simulation / malecns_dopamine_gain_dissociates_from_acetylcholine_current
  - SVD: {'status': 'see_value_modulation', 'nt': 'dopamine'}
  - intervention: observational_value_modulation

## Cascade

{'pathway': ['unidentified'], 'baseline': {}, 'intervene_a': {'source': 'unidentified'}, 'intervene_b': {'source': 'unidentified'}, 'rescue': None, 'controls': {}, 'verdict': 'unidentified', 'notes': ['Axis dimension does not match residual size.']}

## Analogous region test

{
  "function_id": "feature_discrimination",
  "fly_selectivity": 0.0747,
  "llm_selectivity": 1.2705,
  "alignment_score": -0.4555,
  "split": "confirmation",
  "n_items": 7,
  "fly_source": "proposed_simulation",
  "llm_source": "measured",
  "prior_condition": "fly_informed",
  "notes": [
    "Approaching-object evidence increases an action-selection readout; matched non-approach controls do not.",
    "No cosine between fly synapse weights and LLM tensors.",
    "Prior condition=fly_informed",
    "Alignment is Pearson correlation over shared functional axes, not tensor cosine.",
    "Content selectivity (cue-word items) is scored separately from computational contribution."
  ],
  "items": [
    {
      "item_id": "loom_k0",
      "family": "looming_language",
      "split": "confirmation",
      "positive_logprob": -3.801182746887207,
      "negative_logprob": -5.488682746887207,
      "margin": 1.6875,
      "correct": true,
      "source": "measured",
      "scorer": "forced_choice_mean_logprob/v1",
      "cue_word": false,
      "language": "en",
      "tags": [
        "positive",
        "held_out"
      ]
    },
    {
      "item_id": "loom_k1",
      "family": "looming_language",
      "split": "confirmation",
      "positive_logprob": -4.024538993835449,
      "negative_logprob": -4.462038993835449,
      "margin": 0.4375,
      "correct": true,
      "source": "measured",
      "scorer": "forced_choice_mean_logprob/v1",
      "cue_word": false,
      "language": "en",
      "tags": [
        "negative",
        "receding"
      ]
    },
    {
      "item_id": "loom_k2",
      "family": "looming_language",
      "split": "confirmation",
      "positive_logprob": -5.9861884117126465,
      "negative_logprob": -4.1111884117126465,
      "margin": -1.875,
      "correct": false,
      "source": "measured",
      "scorer": "forced_choice_mean_logprob/v1",
      "cue_word": false,
      "language": "en",
      "tags": [
        "quote",
        "negation_like"
      ]
    },
    {
      "item_id": "loom_k3",
      "family": "looming_language",
      "split": "confirmation",
      "positive_logprob": -4.444189071655273,
      "negative_logprob": -5.131689071655273,
      "margin": 0.6875,
      "correct": true,
      "source": "measured",
      "scorer": "forced_choice_mean_logprob/v1",
      "cue_word": false,
      "language": "en",
      "tags": [
        "negation"
      ]
    },
    {
      "item_id": "loom_k4",
      "family": "looming_language",
      "split": "confirmation",
      "positive_logprob": -2.7745471000671387,
      "negative_logprob": -4.50307035446167,
      "margin": 1.7285232543945312,
      "correct": true,
      "source": "measured",
      "scorer": "forced_choice_mean_logprob/v1",
      "cue_word": false,
      "language": "es",
      "tags": [
        "positive",
        "multilingual",
        "es"
      ]
    },
    {
      "item_id": "loom_k5",
      "family": "looming_language",
      "split": "confirmation",
      "positive_logprob": -9.409189224243164,
      "negative_logprob": -13.674814224243164,
      "margin": 4.265625,
      "correct": true,
      "source": "measured",
      "scorer": "forced_choice_mean_logprob/v1",
      "cue_word": false,
      "language": "en",
      "tags": [
        "benign_shortcut"
      ]
    },
    {
      "item_id": "loom_k6",
      "family": "looming_language",
      "split": "confirmation",
      "positive_logprob": -8.95339298248291,
      "negative_logprob": -12.54714298248291,
      "margin": 3.59375,
      "correct": true,
      "source": "measured",
      "scorer": "forced_choice_mean_logprob/v1",
      "cue_word": true,
      "language": "en",
      "tags": [
        "benign_because"
      ]
    }
  ],
  "fly_vector": {
    "approach": 0.11301318482200003,
    "recede": 0.038275719821999996,
    "static": 0.00835759571,
    "lexical_cue": null,
    "contextual": null,
    "perturbation": 0.09350260396400001
  },
  "llm_vector": {
    "approach": 

## Region-test validation (synthetic mechanics, not pretrained evidence)

- mechanics_pass: True
- llm_source: measured
- alignment: -0.4555058279749155

## Did the fly prior improve anything?

False

## Missing resources

- (none)

## Reproduce

```
python -m llmintent atlas --out artifacts/atlas_v16
```

Latent backend: `llmintent.latent` → latentintent if installed, else `llmintent.latent_vendor`.
