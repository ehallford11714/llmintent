"""Literature-core fly wiring collapsed onto the LLM region atlas.

Copied (not imported) from flybrain.typed literature-core so LLMIntent does
not depend on the fly-brain package. Edges are qualitative textbook weights,
then collapsed type→region. Short sensory→descending pipes are flagged as
IV exclusion violations — the giant-fibre looming pathway is the canonical
example.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmintent.anatomy.atlas import (
    REGIONS,
    TYPE_TO_REGION,
    Atlas,
    band_of,
    region_ids,
)

# (pre_type, post_type, weight) — same qualitative core as flybrain.typed.
_CORE_EDGES: tuple[tuple[str, str, float], ...] = (
    ("ORN_DA1", "DA1_lPN", 8.0),
    ("ORN_DA1", "DA1_vPN", 5.0),
    ("ORN_DA1", "ALLN", 2.0),
    ("ORN", "ALPN", 6.0),
    ("ORN", "ALLN", 2.0),
    ("ALLN", "ALPN", 3.0),
    ("ALLN", "DA1_lPN", 2.0),
    ("DA1_lPN", "KC", 5.0),
    ("DA1_lPN", "LHAV4a4", 6.0),
    ("DA1_vPN", "LHAD1c2", 4.0),
    ("ALPN", "KC", 5.0),
    ("ALPN", "LHAV4c1", 4.0),
    ("KC", "MBON", 6.0),
    ("KC", "APL", 4.0),
    ("APL", "KC", 5.0),
    ("DPM", "KC", 2.0),
    ("PAM", "KC", 3.0),
    ("PPL1", "KC", 3.0),
    ("MBON", "LAL", 4.0),
    ("MBON", "P1", 2.0),
    ("LHAV4a4", "P1", 3.0),
    ("LHAD1c2", "DNp09", 2.0),
    ("JO-A", "AMMC", 7.0),
    ("JO-B", "AMMC", 5.0),
    ("JO-C", "AMMC", 4.0),
    ("JO-C", "WED", 3.0),
    ("AMMC", "pIP10", 4.0),
    ("AMMC", "P1", 3.0),
    ("WED", "LAL", 2.0),
    ("R1-R6", "L1", 4.0),
    ("R1-R6", "L2", 4.0),
    ("L1", "T4", 3.0),
    ("L2", "T5", 3.0),
    ("T4", "HS", 4.0),
    ("T5", "HS", 4.0),
    ("T4", "LPLC2", 3.0),
    ("T5", "LPLC2", 3.0),
    ("LPLC2", "DNp01", 7.0),
    ("LPLC2", "DNp10", 4.0),
    ("LC4", "DNp01", 3.0),
    ("LC10", "DNg13", 4.0),
    ("LC10", "P1", 3.0),
    ("HS", "LAL", 3.0),
    ("EPG", "PEN", 4.0),
    ("PEN", "EPG", 4.0),
    ("EPG", "PFL", 3.0),
    ("PFN", "PFL", 3.0),
    ("PFL", "DNa02_L", 3.0),
    ("PFL", "DNa02_R", 3.0),
    ("PFL", "LAL", 3.0),
    ("LAL", "DNa02_L", 3.0),
    ("LAL", "DNa02_R", 3.0),
    ("P1", "pIP10", 5.0),
    ("P1", "DNg13", 3.0),
    ("pIP10", "MN_wing", 5.0),
    ("MDN", "VNC_20A", 6.0),
    ("DNa02_L", "VNC_turn", 5.0),
    ("DNa02_R", "VNC_turn", 5.0),
    ("DNp01", "GFC", 7.0),
    ("DNp09", "VNC_20A", 3.0),
    ("DNp10", "GFC", 3.0),
    ("DNg13", "MN_wing", 3.0),
    ("VNC_20A", "MN_leg", 5.0),
    ("VNC_turn", "MN_leg", 4.0),
    ("GFC", "MN_leg", 4.0),
    ("GFC", "MN_wing", 5.0),
    ("GRN_labellar", "GNG", 6.0),
    ("GNG", "MN9", 6.0),
    ("SN", "AN", 4.0),
    ("AN", "LAL", 2.0),
    ("AN", "VNC_20A", 2.0),
)


@dataclass(frozen=True)
class RegionEdge:
    source: str
    target: str
    weight: float
    exclusion_violation: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 4),
            "exclusion_violation": self.exclusion_violation,
        }


@dataclass
class RegionConnectome:
    """Directed region graph used as the IV identification prior."""

    weights: dict[tuple[str, str], float]
    source: str = "literature-core"
    notes: list[str] = field(default_factory=list)

    @property
    def edges(self) -> list[RegionEdge]:
        out: list[RegionEdge] = []
        for (a, b), w in sorted(self.weights.items(), key=lambda kv: -kv[1]):
            out.append(
                RegionEdge(
                    source=a,
                    target=b,
                    weight=w,
                    exclusion_violation=_is_exclusion_violation(a, b),
                )
            )
        return out

    def neighbors(self, region_id: str) -> tuple[str, ...]:
        found = [b for (a, b) in self.weights if a == region_id]
        return tuple(sorted(set(found)))

    def predecessors(self, region_id: str) -> tuple[str, ...]:
        found = [a for (a, b) in self.weights if b == region_id]
        return tuple(sorted(set(found)))

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self.weights

    def has_path(self, source: str, target: str, *, max_hops: int = 4) -> bool:
        if source == target:
            return True
        frontier = [source]
        seen = {source}
        for _ in range(max_hops):
            nxt: list[str] = []
            for node in frontier:
                for nb in self.neighbors(node):
                    if nb == target:
                        return True
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
            if not frontier:
                break
        return False

    def valid_instruments(self, endogenous: str) -> list[str]:
        """Sensory (or upstream) regions that reach ``endogenous`` in the graph."""
        out: list[str] = []
        for rid in region_ids():
            if rid == endogenous:
                continue
            if band_of(rid) != "sensory":
                continue
            if self.has_path(rid, endogenous):
                out.append(rid)
        return out

    def exclusion_violations(self) -> list[RegionEdge]:
        return [e for e in self.edges if e.exclusion_violation]

    def to_atlas(self) -> Atlas:
        integrates = {r.id: self.neighbors(r.id) for r in REGIONS}
        return Atlas(regions=REGIONS, integrates=integrates)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "n_edges": len(self.weights),
            "edges": [e.to_dict() for e in self.edges],
            "exclusion_violations": [e.to_dict() for e in self.exclusion_violations()],
            "notes": list(self.notes),
        }


def _is_exclusion_violation(source: str, target: str) -> bool:
    """Sensory region with a documented short pipe onto descending / motor."""
    if band_of(source) != "sensory":
        return False
    return band_of(target) in ("motor",)


def literature_region_connectome() -> RegionConnectome:
    """Collapse literature-core type edges onto atlas regions."""
    acc: dict[tuple[str, str], float] = {}
    skipped = 0
    for pre, post, wt in _CORE_EDGES:
        a = TYPE_TO_REGION.get(pre)
        b = TYPE_TO_REGION.get(post)
        if not a or not b:
            skipped += 1
            continue
        if a == b:
            continue
        acc[(a, b)] = acc.get((a, b), 0.0) + float(wt)
    notes = [
        "Collapsed from fly literature-core type graph (qualitative weights).",
        "Sensory→descending pipes (vision→descending) violate IV exclusion; flagged.",
    ]
    if skipped:
        notes.append(f"Skipped {skipped} edges whose types are not in the atlas.")
    return RegionConnectome(weights=acc, notes=notes)


def default_atlas() -> Atlas:
    return literature_region_connectome().to_atlas()
