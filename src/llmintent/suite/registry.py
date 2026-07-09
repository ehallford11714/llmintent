"""Curated HF model registry for Qwen / Mistral / MiniMax / GLM / legacy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

FAMILIES: tuple[str, ...] = ("qwen", "mistral", "minimax", "glm", "legacy")
SIZES: tuple[str, ...] = ("tiny", "small", "medium", "large", "xl")


@dataclass(frozen=True)
class ModelSpec:
    """One curated model entry in the suite registry."""

    family: str
    size: str
    hf_id: str
    params_b: float
    """Approximate total parameters in billions (activated for MoE when noted)."""
    vram_gb_fp16: float
    """Rough FP16/BF16 VRAM for full weights (not KV cache)."""
    description: str
    chat_template: bool = True
    trust_remote_code: bool = True
    gated: bool = False
    moe: bool = False
    active_params_b: float | None = None
    api_alternative: str | None = None
    alternates: tuple[str, ...] = ()
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.family}:{self.size}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        d["alternates"] = list(self.alternates)
        return d


# ---------------------------------------------------------------------------
# Family catalogs (research snapshot: mid-2026)
# Primary IDs prefer widely used instruct / chat checkpoints on Hugging Face.
# ---------------------------------------------------------------------------

_QWEN: tuple[ModelSpec, ...] = (
    ModelSpec(
        family="qwen",
        size="tiny",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        params_b=0.5,
        vram_gb_fp16=1.5,
        description="Qwen2.5 0.5B Instruct — edge / CI-friendly multilingual SLM",
        alternates=("Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-1.5B-Instruct"),
        api_alternative="DashScope / OpenRouter qwen2.5-0.5b",
        notes="Default Live suite target. Qwen3-0.6B is a thinking-capable alternate.",
    ),
    ModelSpec(
        family="qwen",
        size="small",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        params_b=3.0,
        vram_gb_fp16=8.0,
        description="Qwen2.5 3B Instruct — laptop GPU / 8GB class",
        alternates=("Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen2.5-1.5B-Instruct"),
        api_alternative="DashScope qwen-plus / OpenRouter",
    ),
    ModelSpec(
        family="qwen",
        size="medium",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        params_b=7.6,
        vram_gb_fp16=16.0,
        description="Qwen2.5 7B Instruct — strong general instruct baseline",
        alternates=("Qwen/Qwen3-8B", "Qwen/Qwen2.5-14B-Instruct"),
        api_alternative="DashScope qwen-plus",
    ),
    ModelSpec(
        family="qwen",
        size="large",
        hf_id="Qwen/Qwen2.5-32B-Instruct",
        params_b=32.0,
        vram_gb_fp16=64.0,
        description="Qwen2.5 32B Instruct — workstation / multi-GPU",
        alternates=("Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        api_alternative="DashScope qwen-max",
        notes="Qwen3-30B-A3B MoE activates ~3.3B — lower VRAM with recent transformers.",
    ),
    ModelSpec(
        family="qwen",
        size="xl",
        hf_id="Qwen/Qwen2.5-72B-Instruct",
        params_b=72.0,
        vram_gb_fp16=144.0,
        description="Qwen2.5 72B Instruct — cluster / quantized local",
        alternates=(
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "Qwen/Qwen3.6-35B-A3B",
        ),
        api_alternative="DashScope qwen-max / Together / Fireworks",
        notes="Prefer MoE alternates or AWQ/GPTQ for single-node; ModelScope mirror if HF slow.",
    ),
)

_MISTRAL: tuple[ModelSpec, ...] = (
    ModelSpec(
        family="mistral",
        size="tiny",
        hf_id="mistralai/Ministral-3-3B-Instruct-2512",
        params_b=3.0,
        vram_gb_fp16=8.0,
        description="Ministral 3 3B Instruct (2512) — edge multimodal-capable",
        alternates=("mistralai/Ministral-3-3B-Instruct-2512-BF16",),
        api_alternative="Mistral API ministral-3b / Azure / Bedrock",
        notes="Apache 2.0. BF16 variant if default FP8 build is awkward on older GPUs.",
    ),
    ModelSpec(
        family="mistral",
        size="small",
        hf_id="mistralai/Ministral-3-8B-Instruct-2512",
        params_b=8.0,
        vram_gb_fp16=18.0,
        description="Ministral 3 8B Instruct — balanced local instruct",
        alternates=(
            "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ),
        api_alternative="Mistral API ministral-8b-latest",
    ),
    ModelSpec(
        family="mistral",
        size="medium",
        hf_id="mistralai/Ministral-3-14B-Instruct-2512",
        params_b=14.0,
        vram_gb_fp16=30.0,
        description="Ministral 3 14B Instruct — near Small-class quality",
        alternates=("mistralai/Mistral-Nemo-Instruct-2407",),
        api_alternative="Mistral API",
    ),
    ModelSpec(
        family="mistral",
        size="large",
        hf_id="mistralai/Mistral-Small-24B-Instruct-2501",
        params_b=24.0,
        vram_gb_fp16=55.0,
        description="Mistral Small 24B Instruct (2501) — agentic / tool-use",
        alternates=("mistralai/Mixtral-8x7B-Instruct-v0.1",),
        api_alternative="Mistral API mistral-small-latest",
        notes="~55GB BF16; Mixtral-8x7B MoE is a lighter alternate (~45GB).",
    ),
    ModelSpec(
        family="mistral",
        size="xl",
        hf_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
        params_b=141.0,
        vram_gb_fp16=280.0,
        description="Mixtral 8x22B Instruct — sparse MoE flagship open weights",
        moe=True,
        active_params_b=39.0,
        alternates=("mistralai/Mistral-Large-Instruct-2411",),
        api_alternative="Mistral API mistral-large-latest / Large 3 on Studio",
        notes="Mistral Large 3 (675B MoE) may be gated or API-first; check HF org page.",
        gated=False,
    ),
)

_MINIMAX: tuple[ModelSpec, ...] = (
    ModelSpec(
        family="minimax",
        size="tiny",
        hf_id="MiniMaxAI/MiniMax-M2",
        params_b=230.0,
        vram_gb_fp16=80.0,
        description="MiniMax-M2 MoE (~10B active) — smallest practical open MiniMax text MoE",
        moe=True,
        active_params_b=10.0,
        trust_remote_code=True,
        api_alternative="https://api.minimax.io — model MiniMax-M2",
        notes=(
            "MiniMax does not ship sub-7B dense instruct weights. "
            "tiny/small map to M2; prefer API for low-VRAM hosts."
        ),
    ),
    ModelSpec(
        family="minimax",
        size="small",
        hf_id="MiniMaxAI/MiniMax-M2",
        params_b=230.0,
        vram_gb_fp16=80.0,
        description="MiniMax-M2 — coding & agentic MoE (10B active)",
        moe=True,
        active_params_b=10.0,
        api_alternative="api.minimax.io MiniMax-M2 / M2.1",
        notes="Use vLLM / SGLang / transformers with trust_remote_code.",
    ),
    ModelSpec(
        family="minimax",
        size="medium",
        hf_id="MiniMaxAI/MiniMax-M2",
        params_b=230.0,
        vram_gb_fp16=80.0,
        description="MiniMax-M2 — recommended local MiniMax tier",
        moe=True,
        active_params_b=10.0,
        api_alternative="api.minimax.io MiniMax-M2.5 / M2.7",
    ),
    ModelSpec(
        family="minimax",
        size="large",
        hf_id="MiniMaxAI/MiniMax-Text-01",
        params_b=456.0,
        vram_gb_fp16=200.0,
        description="MiniMax-Text-01 — 456B MoE (~46B active), long-context",
        moe=True,
        active_params_b=45.9,
        api_alternative="api.minimax.io (legacy Text-01 endpoints)",
        notes="Requires multi-GPU; trust_remote_code=True.",
    ),
    ModelSpec(
        family="minimax",
        size="xl",
        hf_id="MiniMaxAI/MiniMax-M3",
        params_b=428.0,
        vram_gb_fp16=200.0,
        description="MiniMax-M3 — frontier multimodal MoE (~23B active, 1M context)",
        moe=True,
        active_params_b=23.0,
        api_alternative="api.minimax.io MiniMax-M3",
        notes="Prefer vLLM/SGLang; transformers support evolving (see HF model card).",
    ),
)

_GLM: tuple[ModelSpec, ...] = (
    ModelSpec(
        family="glm",
        size="tiny",
        hf_id="THUDM/chatglm3-6b",
        params_b=6.0,
        vram_gb_fp16=14.0,
        description="ChatGLM3-6B — classic compact GLM chat (legacy path)",
        alternates=("zai-org/glm-4-9b-chat-hf",),
        api_alternative="Z.ai / Zhipu Open Platform glm-4-flash",
        notes="Older ChatGLM stack; prefer GLM-4-9B for new work when VRAM allows.",
    ),
    ModelSpec(
        family="glm",
        size="small",
        hf_id="zai-org/GLM-4-9B-0414",
        params_b=9.0,
        vram_gb_fp16=20.0,
        description="GLM-4-9B-0414 — modern 9B instruct / tool-calling",
        alternates=("THUDM/GLM-4-9B-0414", "zai-org/glm-4-9b-chat-hf"),
        api_alternative="Z.ai glm-4-air / glm-4-flash",
    ),
    ModelSpec(
        family="glm",
        size="medium",
        hf_id="zai-org/GLM-4.7-Flash",
        params_b=30.0,
        vram_gb_fp16=40.0,
        description="GLM-4.7-Flash — 30B-A3B MoE, efficient local flagship",
        moe=True,
        active_params_b=3.0,
        alternates=("zai-org/GLM-4-32B-0414", "THUDM/GLM-4-32B-0414"),
        api_alternative="Z.ai glm-4.7-flash",
        notes="vLLM/SGLang recommended; dense 32B alternate if MoE unsupported.",
    ),
    ModelSpec(
        family="glm",
        size="large",
        hf_id="zai-org/GLM-4.5-Air",
        params_b=106.0,
        vram_gb_fp16=120.0,
        description="GLM-4.5-Air — 106B-A12B MoE reasoning / tools",
        moe=True,
        active_params_b=12.0,
        alternates=("zai-org/GLM-4.5-Air-FP8",),
        api_alternative="Z.ai glm-4.5-air",
    ),
    ModelSpec(
        family="glm",
        size="xl",
        hf_id="zai-org/GLM-4.7",
        params_b=355.0,
        vram_gb_fp16=400.0,
        description="GLM-4.7 — 355B-A32B MoE frontier open weights",
        moe=True,
        active_params_b=32.0,
        alternates=("zai-org/GLM-4.5", "zai-org/GLM-4.6"),
        api_alternative="Z.ai glm-4.7 / glm-4.5",
        notes="Multi-node; FP8 variants cut VRAM. ModelScope mirrors available.",
    ),
)

_LEGACY: tuple[ModelSpec, ...] = (
    ModelSpec(
        family="legacy",
        size="tiny",
        hf_id="distilgpt2",
        params_b=0.082,
        vram_gb_fp16=0.5,
        description="DistilGPT-2 — offline CI / minimal latency baseline",
        chat_template=False,
        trust_remote_code=False,
        notes="Default offline CI path. Never downloads multi-GB weights.",
    ),
    ModelSpec(
        family="legacy",
        size="small",
        hf_id="gpt2",
        params_b=0.124,
        vram_gb_fp16=0.6,
        description="GPT-2 base — classic research / ablation target",
        chat_template=False,
        trust_remote_code=False,
        alternates=("gpt2-medium",),
    ),
    ModelSpec(
        family="legacy",
        size="medium",
        hf_id="gpt2-medium",
        params_b=0.355,
        vram_gb_fp16=1.5,
        description="GPT-2 medium — larger legacy baseline",
        chat_template=False,
        trust_remote_code=False,
        alternates=("facebook/opt-125m",),
    ),
    ModelSpec(
        family="legacy",
        size="large",
        hf_id="gpt2-large",
        params_b=0.774,
        vram_gb_fp16=3.0,
        description="GPT-2 large — legacy upper tier",
        chat_template=False,
        trust_remote_code=False,
    ),
    ModelSpec(
        family="legacy",
        size="xl",
        hf_id="gpt2-xl",
        params_b=1.5,
        vram_gb_fp16=6.0,
        description="GPT-2 XL — largest legacy GPT-2 checkpoint",
        chat_template=False,
        trust_remote_code=False,
    ),
)

_ALL: tuple[ModelSpec, ...] = _QWEN + _MISTRAL + _MINIMAX + _GLM + _LEGACY
_BY_KEY: dict[str, ModelSpec] = {s.key: s for s in _ALL}
_BY_FAMILY: dict[str, dict[str, ModelSpec]] = {}
for _spec in _ALL:
    _BY_FAMILY.setdefault(_spec.family, {})[_spec.size] = _spec


def _norm_family(family: str) -> str:
    f = family.strip().lower()
    aliases = {
        "qwen2": "qwen",
        "qwen2.5": "qwen",
        "qwen3": "qwen",
        "ministral": "mistral",
        "mixtral": "mistral",
        "mini-max": "minimax",
        "minimaxai": "minimax",
        "chatglm": "glm",
        "zhipu": "glm",
        "zai": "glm",
        "gpt2": "legacy",
        "slm": "legacy",
    }
    return aliases.get(f, f)


def _norm_size(size: str) -> str:
    s = size.strip().lower()
    aliases = {
        "xs": "tiny",
        "sm": "small",
        "md": "medium",
        "med": "medium",
        "lg": "large",
        "xxl": "xl",
        "extra-large": "xl",
        "extralarge": "xl",
    }
    return aliases.get(s, s)


def get_model_spec(family: str, size: str = "medium") -> ModelSpec:
    """Return the curated :class:`ModelSpec` for ``family`` + ``size``."""
    fam = _norm_family(family)
    sz = _norm_size(size)
    if fam not in _BY_FAMILY:
        raise KeyError(f"Unknown family {family!r}. Available: {list(FAMILIES)}")
    if sz not in _BY_FAMILY[fam]:
        raise KeyError(
            f"Unknown size {size!r} for family {fam!r}. Available: {list(SIZES)}"
        )
    return _BY_FAMILY[fam][sz]


def list_models(
    family: str | None = None,
    *,
    include_legacy: bool = True,
) -> list[dict]:
    """List registry entries as plain dicts (offline, no downloads)."""
    specs: Iterable[ModelSpec] = _ALL
    if family is not None:
        fam = _norm_family(family)
        specs = [s for s in _ALL if s.family == fam]
    elif not include_legacy:
        specs = [s for s in _ALL if s.family != "legacy"]
    return [s.to_dict() for s in specs]


def list_families() -> list[str]:
    return list(FAMILIES)


def list_sizes() -> list[str]:
    return list(SIZES)
