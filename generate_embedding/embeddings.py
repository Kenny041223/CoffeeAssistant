"""Derived product vectors; menu facts remain in structure.json."""
from typing import Annotated, Literal

from pydantic import Field

from generate_embedding.menu import Nonempty, StrictModel

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBEDDING_DIMENSIONS = 1024
QUERY_INSTRUCTION = "Given a customer's coffee shop request, retrieve relevant menu products."
QUERY_PROMPT = f"Instruct: {QUERY_INSTRUCTION}\nQuery: "

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Vector = Annotated[list[float], Field(min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS)]


class EmbeddingRecipe(StrictModel):
    model: Literal[EMBEDDING_MODEL] = EMBEDDING_MODEL
    revision: Literal[EMBEDDING_REVISION] = EMBEDDING_REVISION
    dimensions: Literal[1024] = EMBEDDING_DIMENSIONS
    dtype: Literal["float32"] = "float32"
    normalized: Literal[True] = True
    similarity: Literal["cosine"] = "cosine"
    document_prompt: Literal[""] = ""
    query_prompt: Literal[QUERY_PROMPT] = QUERY_PROMPT
    max_length: int = Field(ge=1, le=32768)


class ProductEmbedding(StrictModel):
    product_id: Nonempty
    text_sha256: Sha256
    vector: Vector


class MenuEmbeddings(StrictModel):
    schema_version: Literal[1] = 1
    menu_file: Nonempty
    menu_sha256: Sha256
    recipe: EmbeddingRecipe
    created_at: Nonempty
    device: Literal["cuda", "cpu"]
    batch_size: int = Field(gt=0)
    duration_seconds: float = Field(ge=0)
    product_count: int = Field(gt=0)
    products: Annotated[list[ProductEmbedding], Field(min_length=1)]
