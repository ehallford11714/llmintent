"""Unified activation trajectory mapping across layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from llmintent.activation import activation_summary, identify_activation_layers
from llmintent.cognitive.orchestrator import build_cognitive_module_profile
from llmintent.forward import forward_hidden_states
from llmintent.jspace.trace import build_intent_trace
from llmintent.kernels.kl_kernel import per_layer_kl_profile
from llmintent.models import ModelBundle
from llmintent.query.concept_query import ConceptQueryResult, query_concept_in_trajectory
from llmintent.jspace.transport import TransportMaps
from llmintent.poles import build_numerical_pole
from llmintent.steering import get_entropy_trajectory


@dataclass
class TrajectoryMapping:
    """
    Unified layer-by-layer activation trajectory for a prompt (and optional twin).

    Combines entropy maturation, KL tension, intensity, J-space intents,
    cognitive modules, and optional concept-query annotations.
    """

    prompt: str
    twin_b: str | None
    model_name: str
    num_layers: int
    layers: pd.DataFrame
    pivots: dict[str, int] = field(default_factory=dict)
    concept_hits: dict[str, ConceptQueryResult] = field(default_factory=dict)

    def peak_layer(self, metric: str = "concept_activation") -> int | None:
        if metric not in self.layers.columns:
            return None
        return int(self.layers[metric].idxmax())

    def layers_for_concept(self, concept: str) -> list[int]:
        if concept in self.concept_hits:
            return self.concept_hits[concept].matched_layers
        col = f"concept_{concept.replace(' ', '_')}_activation"
        if col in self.layers.columns:
            return self.layers.nlargest(3, col)["layer"].astype(int).tolist()
        return []

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "twin_b": self.twin_b,
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "pivots": self.pivots,
            "layers": self.layers.to_dict(orient="records"),
            "concept_hits": {k: v.to_dict() for k, v in self.concept_hits.items()},
        }


def build_trajectory_mapping(
    bundle: ModelBundle,
    prompt: str,
    *,
    twin_b: str | None = None,
    transport: TransportMaps | None = None,
    position: int = -1,
    concepts: list[str] | None = None,
    include_cognitive: bool = True,
    include_concepts: bool = True,
) -> TrajectoryMapping:
    """
    Build a full activation trajectory map for a prompt.

    Merges:
    - entropy maturation curve
    - numerical pole intensity
    - KL divergence vs twin (if provided)
    - J-space top intent + occupancy per layer
    - cognitive module assignment
    - optional semantic concept query annotations
    """
    twin_b = twin_b or prompt
    trace = build_intent_trace(bundle, prompt, transport=transport, position=position)
    act = activation_summary(bundle, prompt, position=position)
    ent = get_entropy_trajectory(bundle, prompt)

    pole = build_numerical_pole(bundle)
    _, states = forward_hidden_states(bundle, prompt)
    pos = position if position >= 0 else states[0].shape[1] + position

    from llmintent.metrics import cosine_intensity

    intensity = [
        cosine_intensity(states[i][0, pos, :], pole.to(bundle.device))
        for i in range(len(states))
    ]

    traj = pd.DataFrame({"layer": range(len(states))})
    traj = traj.merge(ent.rename(columns={"entropy": "entropy"}), on="layer", how="left")
    traj = traj.merge(act[["layer", "occupancy", "intensity", "entropy_drop"]], on="layer", how="left")
    traj["intensity"] = intensity

    if twin_b != prompt:
        kl, _ = per_layer_kl_profile(bundle, prompt, twin_b, position=position)
        traj["kl_divergence"] = kl.numpy()
        traj["kl_weight"] = kl.numpy() / (kl.max().item() + 1e-8)
    else:
        traj["kl_divergence"] = 0.0
        traj["kl_weight"] = 0.0

    traj["top_intent"] = trace.layer_stats["top1_intent"].values
    traj["top_intent_prob"] = trace.layer_stats["top1_prob"].values
    traj["motor_alignment"] = trace.layer_stats["motor_alignment"].values
    traj["regime"] = trace.layer_stats["regime"].values
    traj["normalized_depth"] = trace.layer_stats["normalized_depth"].values

    cognitive = None
    if include_cognitive and twin_b != prompt:
        cognitive = build_cognitive_module_profile(
            bundle, prompt, twin_b, transport=transport, position=position
        )
        traj = traj.merge(
            cognitive.layer_assignments[
                [
                    "layer",
                    "dominant_module",
                    "barlow_invariance",
                    "barlow_redundancy",
                    "reasoning",
                    "meta_reasoning",
                    "ideation",
                ]
            ],
            on="layer",
            how="left",
        )

    pivots = identify_activation_layers(
        bundle,
        prompt,
        layer_stats=trace.layer_stats,
        entropy=trace.entropy,
        occupancy=trace.occupancy,
        position=position,
    )
    traj["is_activation_pivot"] = traj["layer"].isin(set(pivots.values()))
    traj["pivot_tags"] = traj["layer"].apply(
        lambda layer: ", ".join(k for k, v in pivots.items() if v == layer)
    )

    concept_hits: dict[str, ConceptQueryResult] = {}
    if include_concepts and concepts:
        for concept in concepts:
            hit = query_concept_in_trajectory(
                bundle,
                concept,
                prompt,
                twin_b=twin_b if twin_b != prompt else None,
                position=position,
                cognitive_profile=cognitive,
            )
            concept_hits[concept] = hit
            safe = concept.replace(" ", "_")
            traj = traj.merge(
                hit.trajectory[["layer", "concept_similarity", "concept_activation"]].rename(
                    columns={
                        "concept_similarity": f"concept_{safe}_similarity",
                        "concept_activation": f"concept_{safe}_activation",
                    }
                ),
                on="layer",
                how="left",
            )

    return TrajectoryMapping(
        prompt=prompt,
        twin_b=twin_b if twin_b != prompt else None,
        model_name=bundle.name,
        num_layers=len(states),
        layers=traj,
        pivots=pivots,
        concept_hits=concept_hits,
    )
