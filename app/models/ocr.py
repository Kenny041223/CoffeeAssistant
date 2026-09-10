"""Validated OCR evidence; these records are not menu database items."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OCRRegion(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    text: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    # Four corners in pixels of the EXIF-oriented image, starting at top-left.
    box: tuple[tuple[float, float], tuple[float, float],
               tuple[float, float], tuple[float, float]]
    needs_review: bool


class OCRDocument(BaseModel):
    schema_version: int = 1
    source_image: str
    source_sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    engine: str
    review_threshold: float = Field(ge=0, le=1)
    status: Literal["text_detected", "no_text"]
    regions: list[OCRRegion]
    warnings: list[str] = Field(default_factory=list)
