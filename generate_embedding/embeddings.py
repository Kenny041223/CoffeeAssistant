"""Derived product vectors; menu facts remain in structure.json."""
from typing import Annotated, Literal

from pydantic import Field

from generate_embedding.menu import Nonempty, StrictModel

EMBEDDING_MODEL = "gemini-embedding-001"
# Native output is 3072-dim; truncated via Matryoshka Representation Learning.
# Google's own docs: only the untruncated 3072-dim output is pre-normalized --
# any other output_dimensionality needs client-side normalization, which
# GeminiEmbedder always does regardless of dimension, for that reason.
EMBEDDING_DIMENSIONS = 768
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Vector = Annotated[list[float], Field(min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS)]


class EmbeddingRecipe(StrictModel):
    model: Literal[EMBEDDING_MODEL] = EMBEDDING_MODEL
    dimensions: Literal[EMBEDDING_DIMENSIONS] = EMBEDDING_DIMENSIONS
    normalized: Literal[True] = True
    similarity: Literal["cosine"] = "cosine"
    document_task_type: Literal[DOCUMENT_TASK_TYPE] = DOCUMENT_TASK_TYPE
    query_task_type: Literal[QUERY_TASK_TYPE] = QUERY_TASK_TYPE
    max_length: int = Field(ge=1, le=32768)


class ProductEmbedding(StrictModel):
    product_id: Nonempty
    text_sha256: Sha256
    vector: Vector


class MenuEmbeddings(StrictModel):
    schema_version: Literal[2] = 2
    menu_file: Nonempty
    menu_sha256: Sha256
    recipe: EmbeddingRecipe
    created_at: Nonempty
    provider: Literal["gemini"] = "gemini"
    batch_size: int = Field(gt=0)
    duration_seconds: float = Field(ge=0)
    product_count: int = Field(gt=0)
    products: Annotated[list[ProductEmbedding], Field(min_length=1)]
