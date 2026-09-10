"""Qwen transcription evidence, without fabricated confidence or boxes."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class QwenReading(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    transcription: str
    uncertainties: list[str]


class QwenDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2] = 2
    source_image: str
    source_sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    engine: Literal["ollama"] = "ollama"
    model: str
    model_digest: str
    prompt_version: str = "qwen-transcription-v1"
    status: Literal["text_detected", "no_text"]
    transcription: str
    uncertainties: list[str]
    verified: Literal[False] = False
    duration_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=lambda: [
        "Model-generated transcription; not independently verified.",
        "Table layout and decimal prices can be wrong even when no uncertainty is reported.",
        "No calibrated confidence scores or measured bounding boxes are available.",
    ])
