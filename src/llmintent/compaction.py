"""Compaction analysis via SVD + SSO (notebook CompactionAnalyzer)."""

from __future__ import annotations

import gc
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from llmintent.embeddings import EmbeddingSpace
from llmintent.metrics import calculate_sso_score
from llmintent.models import get_ffn_weight, get_input_embeddings, get_transformer_layers
from llmintent.projection import build_projection_matrix
from llmintent.svd import perform_svd_on_ffn


class CompactionAnalyzer:
    """Analyze semantic compaction across FFN layers using SSO isolates."""

    def __init__(
        self,
        model_name: str,
        embedding_space: EmbeddingSpace,
        *,
        device: str | None = None,
        trust_remote_code: bool = True,
        sso_threshold: float = 0.7,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.embedding_space = embedding_space
        self.sso_threshold = sso_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,
        ).cpu()
        self.model.eval()

        self.sem_pole = torch.from_numpy(embedding_space.mean_vector(limit=10_000))
        gram_tokens = [".", ",", "and", "but", "in", "of", "to", "the"]
        gram_vecs = [
            embedding_space.vectors[embedding_space.index(t)]
            for t in gram_tokens
            if embedding_space.index(t) is not None
        ]
        self.gram_pole = torch.from_numpy(__import__("numpy").array(gram_vecs).mean(axis=0))
        self.projection = build_projection_matrix(
            self.model,
            self.tokenizer,
            embedding_space,
        )

    def analyze_compaction(self, top_k: int = 50) -> pd.DataFrame:
        """Return per-layer isolate density and average SSO purity."""
        results: list[dict[str, float | int]] = []
        layers = get_transformer_layers(self.model)

        with torch.no_grad():
            for layer_idx, layer in enumerate(layers):
                w = get_ffn_weight(layer).to(torch.float32)
                top_v = perform_svd_on_ffn(w, top_k=top_k)
                projected = top_v.t() @ self.projection

                for comp_idx in range(projected.shape[0]):
                    vec = projected[comp_idx].unsqueeze(0)
                    sem_sim = abs(F.cosine_similarity(vec, self.sem_pole.unsqueeze(0)).item())
                    str_sim = abs(F.cosine_similarity(vec, self.gram_pole.unsqueeze(0)).item())
                    sso = calculate_sso_score(sem_sim, str_sim)
                    if sso > self.sso_threshold:
                        results.append(
                            {
                                "layer": layer_idx + 1,
                                "component": comp_idx + 1,
                                "sso": sso,
                                "sem_sim": sem_sim,
                                "str_sim": str_sim,
                            }
                        )

        df = pd.DataFrame(results)
        if df.empty:
            return pd.DataFrame(columns=["isolate_density", "avg_purity"])
        summary = (
            df.groupby("layer")["sso"]
            .agg(isolate_density="count", avg_purity="mean")
            .reset_index()
        )
        return summary

    def find_inference_pivot(self, compaction_df: pd.DataFrame) -> int | None:
        """Identify pivot layer from handover gradient of structural density."""
        if compaction_df.empty or len(compaction_df) < 2:
            return None
        work = compaction_df.sort_values("layer").copy()
        work["structural_density"] = 1 - (work["isolate_density"] / work["isolate_density"].max())
        work["handover_gradient"] = work["structural_density"].diff().fillna(0)
        idx = work["handover_gradient"].abs().idxmax()
        return int(work.loc[idx, "layer"])

    def cleanup(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
