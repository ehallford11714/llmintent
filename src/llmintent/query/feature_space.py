"""Build KL + twin-Barlow fused feature space for layer trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from llmintent.forward import forward_hidden_states, normalize_hidden
from llmintent.kernels.barlow import minimize_twin_barlow, per_layer_barlow_features
from llmintent.kernels.kl_kernel import collect_twin_hidden_matrix, per_layer_kl_profile
from llmintent.models import ModelBundle, get_input_embeddings


def build_trajectory_feature_space(
    bundle: ModelBundle,
    twin_a: str,
    twin_b: str,
    *,
    position: int = -1,
    proj_dim: int = 32,
    barlow_steps: int = 60,
) -> tuple[np.ndarray, pd.DataFrame, torch.Tensor]:
    """
    Construct per-layer feature vectors: KL-weighted Barlow projection + twin stats.

    Returns:
        features [n_layers, feat_dim] (L2-normalized rows)
        metadata DataFrame per layer
        barlow projector [hidden, proj_dim]
    """
    kl_profile, _ = per_layer_kl_profile(bundle, twin_a, twin_b, position=position)
    h_a, h_b = collect_twin_hidden_matrix(bundle, twin_a, twin_b, position=position)
    projector, barlow_metrics = minimize_twin_barlow(
        h_a, h_b, kl_profile, proj_dim=proj_dim, steps=barlow_steps
    )
    barlow_diag, barlow_off = per_layer_barlow_features(h_a, h_b)

    h_mean = (h_a + h_b) / 2.0
    h_proj = h_mean @ projector  # [layers, proj_dim]

    kl_norm = kl_profile / (kl_profile.max() + 1e-8)
    kl_amp = (1.0 + kl_norm).unsqueeze(-1)
    barlow_block = h_proj * kl_amp

    stat_block = torch.stack(
        [kl_norm, barlow_diag, barlow_off],
        dim=1,
    )
    combined = torch.cat([barlow_block, stat_block], dim=1).numpy()

    # L2-normalize each layer feature row for cosine KNN
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    features = combined / np.clip(norms, 1e-8, None)

    meta = pd.DataFrame(
        {
            "layer": list(range(len(kl_profile))),
            "kl_divergence": kl_profile.numpy(),
            "kl_weight": kl_norm.numpy(),
            "barlow_invariance": barlow_diag.numpy(),
            "barlow_redundancy": barlow_off.numpy(),
            "barlow_loss": barlow_metrics.get("barlow_loss", 0.0),
        }
    )
    return features, meta, projector


def semantic_concept_vector(
    bundle: ModelBundle,
    concept_text: str,
    projector: torch.Tensor,
    *,
    anchor_prompt: str | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Embed semantic concept text into the same KL-Barlow feature space.

    Blends token embedding mean with contextual hidden state when anchor_prompt given.
    """
    tokenizer = bundle.tokenizer
    token_ids = tokenizer.encode(concept_text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Could not tokenize concept: {concept_text!r}")

    embeddings = get_input_embeddings(bundle.model)
    embed_mean = embeddings[token_ids].mean(dim=0).float().cpu()

    if anchor_prompt:
        anchor = f"{anchor_prompt} [CONCEPT:{concept_text}]"
        _, states = forward_hidden_states(bundle, anchor)
        contextual = normalize_hidden(bundle, states[-1][0, -1, :].float().cpu())
        h_blend = 0.7 * contextual + 0.3 * embed_mean
    else:
        _, states = forward_hidden_states(bundle, concept_text)
        contextual = normalize_hidden(bundle, states[-1][0, -1, :].float().cpu())
        h_blend = 0.5 * contextual + 0.5 * embed_mean

    proj_dim = projector.shape[1]
    h_proj = (h_blend @ projector).numpy()

    # Neutral twin-stat dims (concept query point — no twin diff at concept origin)
    stat_zeros = np.zeros(3, dtype=np.float32)
    vec = np.concatenate([h_proj, stat_zeros])
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm

    diagnostics = {
        "num_concept_tokens": float(len(token_ids)),
        "embed_norm": float(torch.norm(embed_mean).item()),
        "blend_norm": float(torch.norm(h_blend).item()),
        "proj_dim": float(proj_dim),
    }
    return vec, diagnostics
