"""FastAPI server for LLMIntent Live real-time inference."""

from __future__ import annotations

from typing import Any

from llmintent.live.pipeline import LiveIntentPipeline
from llmintent.live.registry import list_live_models
from llmintent.live.session import LiveSessionConfig


def create_app(default_model: str = "qwen-0.5b"):
    """Build FastAPI app. Requires `pip install llmintent[live]`."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "Live API requires fastapi. Install with: pip install llmintent[live]"
        ) from exc

    app = FastAPI(
        title="LLMIntent Live",
        description="Real-time focused reasoning on Phi-3, Qwen 0.5B, and SLMs",
        version="0.9.0",
    )
    pipeline = LiveIntentPipeline(LiveSessionConfig(model_key=default_model))

    class LoadRequest(BaseModel):
        model_key: str

    class AnalyzeRequest(BaseModel):
        prompt: str
        concepts: list[str] | None = None
        include_focus: bool = True

    class HeightenRequest(BaseModel):
        prompt: str
        anchor_prompt: str | None = None
        concepts: list[str] | None = None
        mode: str = "explicit_retrace"
        steer: bool = False

    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = Field(64, ge=1, le=512)
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        retracement_mode: str | None = None
        steer: bool = False
        anchor_prompt: str | None = None

    class ProbeRequest(BaseModel):
        prompt: str
        k: int = Field(5, ge=1, le=20)
        retracement_mode: str | None = None

    @app.get("/models")
    def get_models() -> dict[str, Any]:
        return {"models": list_live_models(), "loaded": pipeline.model_key}

    @app.post("/load")
    def post_load(body: LoadRequest) -> dict[str, Any]:
        pipeline.load(body.model_key)
        return {"loaded": pipeline.model_key, "spec": pipeline.session.spec.to_dict()}

    @app.post("/unload")
    def post_unload() -> dict[str, str]:
        pipeline.unload()
        return {"status": "unloaded"}

    @app.post("/analyze")
    def post_analyze(body: AnalyzeRequest) -> dict[str, Any]:
        return pipeline.analyze(
            body.prompt,
            concepts=body.concepts,
            include_focus=body.include_focus,
        ).to_dict()

    @app.post("/heighten")
    def post_heighten(body: HeightenRequest) -> dict[str, Any]:
        return pipeline.heighten(
            body.prompt,
            anchor_prompt=body.anchor_prompt,
            concepts=body.concepts,
            mode=body.mode,
            steer=body.steer,
        ).to_dict()

    @app.post("/generate")
    def post_generate(body: GenerateRequest) -> dict[str, Any]:
        return pipeline.generate(
            body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            retracement_mode=body.retracement_mode,
            steer=body.steer,
            anchor_prompt=body.anchor_prompt,
        ).to_dict()

    @app.post("/probe")
    def post_probe(body: ProbeRequest) -> dict[str, Any]:
        tokens = pipeline.probe_next_tokens(
            body.prompt,
            k=body.k,
            retracement_mode=body.retracement_mode,
        )
        return {"tokens": [{"token": t, "prob": p} for t, p in tokens]}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": pipeline.model_key}

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, model: str = "qwen-0.5b") -> None:
    """Run uvicorn server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Live serve requires uvicorn. Install with: pip install llmintent[live]"
        ) from exc
    app = create_app(default_model=model)
    uvicorn.run(app, host=host, port=port)
