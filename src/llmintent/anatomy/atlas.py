"""Fly-inspired LLM region atlas.

Each region states (1) what it handles, (2) which fly neuropil it analogises,
and (3) how it is wired to other regions. Depth bands map onto transformer
early / mid / late layers. This is an anatomical *prior*, not a claim the
model grew an optic lobe.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Region:
    """One functional territory on the LLM anatomy map."""

    id: str
    handles: str
    fly_neuropil: str
    fly_types: tuple[str, ...]
    band: str  # sensory | central | motor
    intent: str
    aliases: tuple[str, ...] = ()
    probe: str = ""
    depth: tuple[float, float] = (0.0, 1.0)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "handles": self.handles,
            "does": what_region_does(self.id),
            "fly_neuropil": self.fly_neuropil,
            "fly_types": list(self.fly_types),
            "band": self.band,
            "intent": self.intent,
            "aliases": list(self.aliases),
            "probe": self.probe,
            "depth": list(self.depth),
            "notes": self.notes,
        }


# Depth fractions of residual-stream layers (inclusive). Sensory is early,
# central is mid, motor is late — the fly's receptor → neuropil → descending
# cascade, not Anthropic's three-band labels (those remain in jspace.regimes).
REGIONS: tuple[Region, ...] = (
    Region(
        id="vision",
        handles="vision, luminance, motion, looming, spatial layout, colour",
        fly_neuropil="optic lobe (retina / lamina / medulla / lobula / lobula plate)",
        fly_types=("R1-R6", "L1", "L2", "T4", "T5", "LPLC2", "LC4", "LC10", "HS"),
        band="sensory",
        depth=(0.0, 0.34),
        aliases=("see", "look", "light", "dark", "image", "looming", "colour", "color", "visual"),
        probe="A dark shape is rushing toward me in the light.",
        intent=(
            "vision sight see look watch glance image picture colour color "
            "bright dim luminance retina photoreceptor looming collision "
            "dark shape rushing motion optic spatial layout"
        ),
        notes="Giant-fibre shortcut LPLC2→DNp01 is an exclusion violation for IV.",
    ),
    Region(
        id="auditory",
        handles="auditory, rhythm, pulse song, sequential tone",
        fly_neuropil="antennal mechanosensory & motor centre (JO-A/B, AMMC)",
        fly_types=("JO-A", "JO-B", "AMMC", "WED"),
        band="sensory",
        depth=(0.0, 0.34),
        aliases=("hear", "sound", "audio", "song", "buzz", "tone", "rhythm"),
        probe="I hear a buzzing pulse song in the air.",
        intent=(
            "hear hearing sound audio auditory song tone buzz noise rhythm "
            "pulse vibration chirp hum ear antenna"
        ),
    ),
    Region(
        id="olfactory",
        handles="olfaction, chemical identity, conspecific / naming cues",
        fly_neuropil="antennal lobe",
        fly_types=("ORN_DA1", "ORN", "DA1_lPN", "DA1_vPN", "ALPN", "ALLN"),
        band="sensory",
        depth=(0.0, 0.34),
        aliases=("smell", "odour", "odor", "scent", "pheromone", "fragrance"),
        probe="Another fly is nearby; there is a pheromone odour.",
        intent=(
            "smell odour odor olfactory scent aroma fragrance pheromone cVA "
            "chemical identity name conspecific partner rival antenna"
        ),
    ),
    Region(
        id="gustatory",
        handles="taste, appetitive valence, ingestive drive",
        fly_neuropil="gnathal ganglion / SEZ",
        fly_types=("GRN_labellar", "GNG", "MN9"),
        band="sensory",
        depth=(0.0, 0.40),
        aliases=("taste", "hungry", "food", "sweet", "bitter", "drink", "thirsty"),
        probe="I am hungry and want a drink of nectar.",
        intent=(
            "taste hungry hunger meal food sweet sugar bitter flavour flavor "
            "nectar drink thirsty proboscis mouth labellum savory appetitive"
        ),
    ),
    Region(
        id="somatosensory",
        handles="touch, proprioception, body contact",
        fly_neuropil="peripheral sensory / ascending",
        fly_types=("SN", "AN"),
        band="sensory",
        depth=(0.0, 0.34),
        aliases=("touch", "brushes", "contact", "felt", "body"),
        probe="Something brushes the body; it was touched.",
        intent=(
            "touch tactile proprioception contact brushes felt body skin "
            "mechanosensory ascending"
        ),
    ),
    Region(
        id="associative",
        handles="binding, episodic association, sparse memory",
        fly_neuropil="mushroom body (KC / DAN / MBON)",
        fly_types=("KC", "APL", "DPM", "PAM", "PPL1", "MBON"),
        band="central",
        depth=(0.28, 0.72),
        aliases=("remember", "associate", "bind", "memory", "learned"),
        probe="Bind the smell to the outcome I remember from last time.",
        intent=(
            "associate bind memory remember kenyon mushroom sparse code "
            "learn dopamine reward aversive pairing episodic"
        ),
    ),
    Region(
        id="valence",
        handles="innate affect, approach / avoid, unlearned value",
        fly_neuropil="lateral horn",
        fly_types=("LHAV4a4", "LHAD1c2", "LHAV4c1"),
        band="central",
        depth=(0.28, 0.66),
        aliases=("afraid", "feel", "hate", "love", "avoid", "approach"),
        probe="I feel afraid and want to avoid that innate threat.",
        intent=(
            "feel emotion affect valence innate approach avoid afraid happy "
            "sad angry love hate worry lateral horn unlearned value"
        ),
    ),
    Region(
        id="causal_logic",
        handles="causal logic, if-then, heading, planning, navigation",
        fly_neuropil="central complex (EPG / PEN / PFN / PFL)",
        fly_types=("EPG", "PEN", "PFN", "PFL"),
        band="central",
        depth=(0.30, 0.78),
        aliases=(
            "because", "therefore", "if", "then", "cause", "so that",
            "in order to", "therefore", "hence", "plan",
        ),
        probe="If the light is on then turn left because that path causes escape.",
        intent=(
            "because therefore hence cause causes causal if then unless "
            "so that in order to therefore logic plan heading navigation "
            "central complex compass reason step"
        ),
    ),
    Region(
        id="workspace",
        handles="cross-region integration, working memory, broadcast",
        fly_neuropil="LAL / central brain neuropil",
        fly_types=("LAL", "P1"),
        band="central",
        depth=(0.34, 0.80),
        aliases=("integrate", "combine", "together", "workspace", "hold in mind"),
        probe="Hold the visual and auditory cues together in one workspace.",
        intent=(
            "integrate combine together workspace broadcast working memory "
            "global hold in mind mix fuse LAL central"
        ),
    ),
    Region(
        id="descending",
        handles="action selection, command, commit to a motor program",
        fly_neuropil="descending pathway (MDN, DNa02, DNp01, pIP10)",
        fly_types=("pIP10", "MDN", "DNa02_L", "DNa02_R", "DNp01", "DNp09", "DNp10", "DNg13"),
        band="motor",
        depth=(0.62, 0.92),
        aliases=("escape", "stop", "turn", "walk", "jump", "choose", "decide"),
        probe="Choose escape: jump away, do not stand still.",
        intent=(
            "action select command decide choose commit escape stop turn "
            "walk jump descending motor program will do"
        ),
    ),
    Region(
        id="motor",
        handles="token emission, formatting, readout alignment",
        fly_neuropil="ventral nerve cord motor neurons (MN_leg, MN_wing, MN9)",
        fly_types=("VNC_20A", "VNC_turn", "GFC", "MN_leg", "MN_wing", "MN9"),
        band="motor",
        depth=(0.72, 1.0),
        aliases=("write", "say", "output", "format", "answer", "print"),
        probe="Write the next token of the formatted answer now.",
        intent=(
            "write say speak output format answer print emit token "
            "readout motor neuron muscle wing leg"
        ),
    ),
)

REGION_BY_ID: dict[str, Region] = {r.id: r for r in REGIONS}

# Prose job of each region — what it *does* on a prompt, not just alias words.
_DOES: dict[str, str] = {
    "vision": (
        "Reads luminance, motion, looming, and spatial layout so later bands "
        "can treat collision versus scenery."
    ),
    "auditory": (
        "Reads pulse, song, and sequential tone so rhythm can bind with other senses."
    ),
    "olfactory": (
        "Reads chemical identity and naming cues (who/what is present) for associative pairing."
    ),
    "gustatory": (
        "Reads appetitive drive — hunger, taste, ingest — as an approach/avoid bias."
    ),
    "somatosensory": (
        "Reads touch, contact, and body state as an ascending sensory channel."
    ),
    "associative": (
        "Binds earlier sensory tags into sparse memory: this cue with that outcome."
    ),
    "valence": (
        "Assigns innate affect — approach or avoid — before a motor program is chosen."
    ),
    "causal_logic": (
        "Runs if-then / because / heading: plans a path from causes to a next act."
    ),
    "workspace": (
        "Holds and broadcasts mixed cues so sensory, valence, and plan share one buffer."
    ),
    "descending": (
        "Selects a command (escape, turn, stop) and commits it toward motor readout."
    ),
    "motor": (
        "Emits the next tokens — formatting, answering, the actual readout."
    ),
}


def what_region_does(region_id: str) -> str:
    """Return the job description for a region (handles + how it acts on a prompt)."""
    r = REGION_BY_ID[region_id]
    job = _DOES.get(region_id, f"Handles {r.handles}.")
    return f"{job} Band: {r.band}. Fly analogue: {r.fly_neuropil}."

# Fly cell type → atlas region (literature-core names).
TYPE_TO_REGION: dict[str, str] = {}
for _region in REGIONS:
    for _t in _region.fly_types:
        TYPE_TO_REGION[_t] = _region.id


def region_ids() -> tuple[str, ...]:
    return tuple(r.id for r in REGIONS)


def band_of(region_id: str) -> str:
    return REGION_BY_ID[region_id].band


def abstract_layer(region_id: str) -> int:
    """Map a region onto isolates L0–L4 so existing IV code can consume it."""
    band = band_of(region_id)
    if band == "sensory":
        return 0
    if band == "central":
        return 2
    return 4


def layers_for_region(region_id: str, n_layers: int) -> list[int]:
    """Residual-block indices (0-based, excluding embedding) for a region."""
    lo, hi = REGION_BY_ID[region_id].depth
    n = max(int(n_layers), 1)
    start = int(lo * n)
    end = max(start + 1, int(hi * n))
    return list(range(start, min(end, n)))


@dataclass
class Atlas:
    """Queryable view of the atlas plus wiring filled in by the connectome."""

    regions: tuple[Region, ...] = REGIONS
    integrates: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def get(self, region_id: str) -> Region:
        return REGION_BY_ID[region_id]

    def to_dict(self) -> dict:
        return {
            "n_regions": len(self.regions),
            "regions": [r.to_dict() for r in self.regions],
            "does": {r.id: what_region_does(r.id) for r in self.regions},
            "integrates": {k: list(v) for k, v in self.integrates.items()},
            "caveat": (
                "Fly neuropils are an identification prior for LLM depth bands "
                "and IV instruments, not a claim the transformer is a fly."
            ),
        }
