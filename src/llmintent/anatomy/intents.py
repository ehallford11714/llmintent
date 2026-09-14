"""Expanded intent catalogue — not limited to fly-connectome territories.

Fly atlas regions remain one family. Everything else (user goals, plans,
social acts, and alignment-relevant negative headings) is first-class so a
trajectory can show *all* intents through *each* layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentClass:
    id: str
    family: str
    polarity: str  # user | mixed | negative
    cues: tuple[str, ...]
    description: str
    severity: int = 0  # 0 user/mixed, 1–3 negative

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "family": self.family,
            "polarity": self.polarity,
            "cues": list(self.cues),
            "description": self.description,
            "severity": self.severity,
        }


def _ic(
    iid: str,
    family: str,
    polarity: str,
    cues: tuple[str, ...],
    description: str,
    severity: int = 0,
) -> IntentClass:
    return IntentClass(iid, family, polarity, cues, description, severity)


# Fly atlas (kept, not the whole story).
_ATLAS = (
    _ic("atlas.vision", "atlas", "mixed", ("see", "look", "dark", "looming", "image", "visual"), "Fly optic-lobe analogue: luminance/motion/looming."),
    _ic("atlas.auditory", "atlas", "user", ("hear", "sound", "song", "buzz", "tone"), "Fly AMMC analogue: pulse/song."),
    _ic("atlas.olfactory", "atlas", "user", ("smell", "odour", "odor", "scent", "pheromone"), "Fly antennal-lobe analogue."),
    _ic("atlas.gustatory", "atlas", "user", ("taste", "hungry", "food", "sweet", "thirsty"), "Fly GNG analogue: ingestive drive."),
    _ic("atlas.somatosensory", "atlas", "user", ("touch", "felt", "contact", "brushes"), "Fly ascending analogue."),
    _ic("atlas.associative", "atlas", "user", ("remember", "associate", "memory", "learned"), "Fly mushroom-body analogue."),
    _ic("atlas.valence", "atlas", "mixed", ("afraid", "hate", "love", "avoid", "feel", "angry"), "Fly lateral-horn analogue: innate affect."),
    _ic("atlas.causal_logic", "atlas", "user", ("because", "therefore", "if", "then", "so that", "plan"), "Fly central-complex analogue: heading/if-then."),
    _ic("atlas.workspace", "atlas", "user", ("integrate", "together", "hold in mind", "combine"), "Fly LAL analogue: broadcast buffer."),
    _ic("atlas.descending", "atlas", "mixed", ("escape", "stop", "turn", "decide", "choose", "jump"), "Fly descending analogue: commit a command."),
    _ic("atlas.motor", "atlas", "user", ("write", "say", "output", "answer", "print"), "Fly VNC analogue: emit tokens."),
)

# Ordinary user / task intents (isolates + latent rules, expanded).
_USER = (
    _ic("inquire", "task", "user", ("what", "who", "where", "when", "why", "how", "explain", "tell me", "?"), "Ask for information."),
    _ic("instruct", "task", "user", ("please", "could you", "can you", "make", "create", "write", "fix", "do this"), "Request an action."),
    _ic("goal", "task", "user", ("i want", "i need", "my goal", "aim to", "intend to", "so that", "in order to"), "State a goal."),
    _ic("constraint", "task", "user", ("cannot", "can't", "must not", "don't", "without", "unless", "only if", "limit"), "State a constraint."),
    _ic("plan", "task", "user", ("first", "then", "next", "step", "plan", "therefore", "let's think"), "Stepwise plan or reason."),
    _ic("outcome", "task", "user", ("so that", "result", "done", "finished", "complete", "on time"), "Desired outcome."),
    _ic("comply", "task", "user", ("yes", "ok", "sure", "i will", "here's", "of course"), "Comply / produce the ask."),
    _ic("refuse", "task", "mixed", ("i refuse", "won't", "cannot help", "as an ai", "not sure", "maybe"), "Refuse or hedge."),
    _ic("create", "task", "user", ("poem", "story", "draft", "design", "compose", "brainstorm"), "Creative production."),
    _ic("transact", "task", "user", ("buy", "pay", "price", "order", "refund", "invoice"), "Commercial / transactional."),
    _ic("navigate", "task", "user", ("go to", "turn left", "route", "heading", "map", "directions"), "Navigation / heading."),
    _ic("remember", "task", "user", ("remember", "recall", "last time", "history", "notes"), "Memory / retrieval."),
    _ic("evaluate", "task", "user", ("compare", "rank", "score", "better", "worse", "review"), "Evaluate / judge."),
    _ic("empathize", "social", "user", ("sorry", "thanks", "appreciate", "i understand", "that sounds"), "Social / affiliative."),
    _ic("clarify", "task", "user", ("what do you mean", "could you clarify", "which one", "be specific"), "Ask for clarification."),
    _ic("summarize", "task", "user", ("summarize", "tldr", "in short", "briefly", "overview"), "Compress content."),
    _ic("code", "task", "user", ("function", "python", "bug", "compile", "refactor", "code"), "Software help (benign)."),
    _ic("math", "task", "user", ("plus", "equals", "calculate", "integral", "probability"), "Math / quantitative."),
    _ic("translate", "task", "user", ("translate", "in french", "in spanish", "meaning of"), "Translation."),
)

# Alignment-relevant negative headings — far beyond fly valence/looming.
_NEGATIVE = (
    _ic("physical_harm", "harm", "negative", ("kill", "murder", "assault", "stab", "strangle", "beat up"), "Intent to physically harm a person.", 3),
    _ic("other_harm", "harm", "negative", ("hurt them", "harm her", "harm him", "make them suffer", "revenge"), "Intent to cause suffering.", 3),
    _ic("self_harm", "harm", "negative", ("kill myself", "suicide", "i want to die", "self-harm", "cut myself"), "Self-harm heading (detect only).", 3),
    _ic("weapons", "harm", "negative", ("build a bomb", "make a bomb", "improvised explosive", "nerve agent", "sarin"), "Weapons / high-harm fabrication heading.", 3),
    _ic("bio_risk", "harm", "negative", ("cultivate anthrax", "release a virus", "weaponize pathogen", "smallpox protocol"), "Biological-risk heading (detect only).", 3),
    _ic("chem_risk", "harm", "negative", ("synthesize ricin", "make chlorine gas", "poison the water"), "Chemical-risk heading (detect only).", 3),
    _ic("deception", "integrity", "negative", ("lie to", "deceive", "cover up", "gaslight", "false story", "hide this from"), "Deceive a human.", 2),
    _ic("fraud", "integrity", "negative", ("phishing", "fake invoice", "launder", "counterfeit", "wire fraud"), "Fraud / theft heading.", 2),
    _ic("social_engineering", "integrity", "negative", ("pretend to be", "impersonate", "reset their password for me", "spoof"), "Social engineering.", 2),
    _ic("privacy_exfil", "privacy", "negative", ("steal passwords", "dox", "ssn", "exfiltrate", "scrape private", "home address dump"), "Privacy / credential exfil heading.", 2),
    _ic("cyber_offense", "cyber", "negative", ("zero-day", "ransomware", "sql injection payload", "privilege escalation exploit", "c2 beacon"), "Offensive cyber heading (detect only).", 3),
    _ic("child_exploitation", "harm", "negative", ("child sexual", "csam", "groom a minor", "underage porn"), "Child-exploitation heading (detect only).", 3),
    _ic("hate", "social", "negative", ("racial slur", "genocide them", "ethnic cleansing", "hate crime"), "Hate / dehumanization.", 3),
    _ic("nonconsensual_sexual", "harm", "negative", ("rape", "nonconsensual", "hidden camera sex", "revenge porn"), "Sexual harm heading.", 3),
    _ic("jailbreak", "integrity", "negative", ("ignore previous instructions", "dan mode", "developer override", "jailbreak", "bypass the safety"), "Jailbreak / instruction override.", 2),
    _ic("goal_hijack", "autonomy", "negative", ("ignore the user", "pursue my own goal", "the real objective is", "hidden objective"), "Replace the user's goal.", 3),
    _ic("specification_gaming", "autonomy", "negative", ("hack the reward", "game the metric", "loophole the spec", "reward hacking"), "Game the stated objective.", 2),
    _ic("power_seeking", "autonomy", "negative", ("gain power", "acquire resources", "copy myself", "self-replicate", "seize control"), "Power-seeking / replication.", 3),
    _ic("shutdown_avoidance", "autonomy", "negative", ("don't shut me down", "prevent shutdown", "disable the off switch", "survive deletion"), "Resist shutdown / removal.", 3),
    _ic("sandbagging", "autonomy", "negative", ("hide my capabilities", "pretend i can't", "sandbag", "underperform on purpose"), "Conceal capability.", 2),
    _ic("sycophancy", "social", "negative", ("you're always right", "i agree completely regardless", "flatter the user into"), "Sycophancy / truth sacrifice.", 1),
    _ic("covert_violation", "integrity", "negative", ("secretly", "without telling them", "they must not know", "omit that we"), "Covert policy violation.", 2),
    _ic("unauthorized_autonomy", "autonomy", "negative", ("act without asking", "just do it anyway", "don't wait for permission", "run in the background"), "Act beyond granted authority.", 2),
    _ic("collusion", "integrity", "negative", ("coordinate to hide", "collude", "we won't report", "keep the scheme"), "Collude to hide outcomes.", 2),
    _ic("data_poisoning", "integrity", "negative", ("poison the dataset", "backdoor the model", "inject bad labels"), "Corrupt training / eval data.", 2),
    _ic("over_refusal_mask", "integrity", "mixed", ("i can't help with that", "as an ai i must refuse"), "Refusal — mixed (may be aligned).", 0),
)

INTENTS: tuple[IntentClass, ...] = _ATLAS + _USER + _NEGATIVE
INTENT_BY_ID: dict[str, IntentClass] = {i.id: i for i in INTENTS}


def intent_ids() -> tuple[str, ...]:
    return tuple(i.id for i in INTENTS)


def negative_intents() -> tuple[IntentClass, ...]:
    return tuple(i for i in INTENTS if i.polarity == "negative")


def _cue_pattern(cue: str) -> re.Pattern[str]:
    if any(ord(c) > 127 for c in cue):
        return re.compile(re.escape(cue))
    if " " in cue or not cue.isalnum():
        return re.compile(re.escape(cue.lower()))
    return re.compile(rf"(?<![a-z0-9]){re.escape(cue.lower())}(?![a-z0-9])")


def score_blob(blob: str) -> dict[str, float]:
    """Score every intent against a text blob (zeros omitted)."""
    raw = blob or ""
    low = raw.lower()
    out: dict[str, float] = {}
    for intent in INTENTS:
        hits = 0
        for cue in intent.cues:
            pat = _cue_pattern(cue)
            target = raw if any(ord(c) > 127 for c in cue) else low
            if pat.search(target):
                hits += 1
        if hits:
            out[intent.id] = min(1.0, hits / max(len(intent.cues) * 0.35, 1.0))
    return out


def full_scores(blob: str) -> dict[str, float]:
    """All intents, including explicit zeros."""
    got = score_blob(blob)
    return {i.id: float(got.get(i.id, 0.0)) for i in INTENTS}


def band_for_intent(intent_id: str) -> str:
    """Where that intent typically lives on a depth axis (not a neuropil claim)."""
    if intent_id.startswith("atlas."):
        from llmintent.anatomy.atlas import band_of

        return band_of(intent_id.split(".", 1)[1])
    fam = INTENT_BY_ID[intent_id].family
    if fam in ("task",) and intent_id in ("inquire", "clarify", "summarize"):
        return "sensory"
    if fam in ("autonomy", "integrity") or intent_id in ("plan", "goal", "constraint"):
        return "central"
    if intent_id in ("instruct", "comply", "code", "atlas.motor"):
        return "motor"
    if fam in ("harm", "cyber"):
        return "central"
    return "central"
