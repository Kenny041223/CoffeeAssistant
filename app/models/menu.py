"""One menu extraction schema, shared across all source layouts."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, BeforeValidator, field_validator

Nonempty = Annotated[str, Field(min_length=1)]
Price = Annotated[float, Field(ge=0)]
OptionalText = Annotated[Nonempty | None, BeforeValidator(
    lambda value: None if isinstance(value, str) and not value.strip() else value)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class MenuVariant(StrictModel):
    size: OptionalText
    temperature: Literal["hot", "iced"] | None
    price: Price | None
    price_note: OptionalText


class MenuItem(StrictModel):
    name: Nonempty
    category: OptionalText
    section: OptionalText
    description: OptionalText
    variants: list[MenuVariant]
    notes: list[str]


class MenuAddon(StrictModel):
    name: Nonempty
    price: Price | None
    applies_to: list[Nonempty] | None

    @field_validator("applies_to", mode="before")
    @classmethod
    def normalize_scope(cls, value):
        if isinstance(value, str):
            return [value] if value.strip() else None
        return value or None


class MenuPage(StrictModel):
    items: list[MenuItem]
    addons: list[MenuAddon]
    notes: list[str]


class MenuSource(StrictModel):
    source_id: str
    ocr_file: str
    ocr_sha256: str
    image_file: str
    image_sha256: str
    ocr_uncertainties: list[str]
    transcription: str
    menu: MenuPage


class StructuredMenu(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["draft"] = "draft"
    model: str
    model_digest: str
    prompt_version: str = "menu-structure-v6"
    currency: str | None
    currency_source: Literal["user_config", "unspecified"]
    source_count: int
    item_occurrence_count: int
    sources: list[MenuSource]
    notes: list[str] = Field(default_factory=lambda: [
        "Structured from OCR transcriptions; original images were not rechecked.",
        "Items are grouped by source. Repeated items across sources are not deduplicated.",
        "Null means unavailable or ambiguous; it does not mean zero or unavailable for sale.",
    ])
