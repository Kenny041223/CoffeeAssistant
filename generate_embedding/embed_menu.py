"""Embed a generated menu with Gemini; no network client connects on import."""
import argparse
import hashlib
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from generate_embedding.embeddings import (
    DOCUMENT_TASK_TYPE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    QUERY_TASK_TYPE,
    EmbeddingRecipe,
    MenuEmbeddings,
)
from generate_embedding.menu import StructuredMenu
from generate_embedding.structure_menu import validate_model_menu


def validate_vectors(vectors: list[list[float]], expected_count: int) -> None:
    """Reject incomplete, malformed, or unnormalized results before publishing."""
    if len(vectors) != expected_count:
        raise ValueError("The embedding model returned the wrong number of vectors")
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Each vector must have {EMBEDDING_DIMENSIONS} dimensions")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
            raise ValueError("Embedding vectors must contain only finite numbers")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("The embedding model returned a zero or unnormalized vector")


def read_menu(path: Path) -> tuple[StructuredMenu, str]:
    raw = path.read_bytes()
    menu = StructuredMenu.model_validate_json(raw)
    if not menu.products:
        raise ValueError("The menu has no products to embed")
    if menu.product_count != len(menu.products) or menu.source_count != len(menu.sources):
        raise ValueError("Menu counts do not match its products or sources")
    ids = [product.product_id for product in menu.products]
    if any(not value.strip() or value != value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Product IDs must be nonempty, unique, and have no surrounding whitespace")
    source_ids = [source.source_id for source in menu.sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("The menu contains duplicate source IDs")
    # Reuse the same identity/evidence checks as generation; never silently merge.
    validate_model_menu(menu, [source.model_dump() for source in menu.sources])
    return menu, hashlib.sha256(raw).hexdigest()


class GeminiEmbedder:
    """One shared client and recipe for both menu documents and future queries.

    Same public interface as the old local QwenEmbedder (encode_documents /
    encode_query) on purpose -- callers like the chatbot don't need to change
    beyond which class they import.
    """

    def __init__(self, *, api_key: str | None = None, batch_size: int = 10, max_length: int = 2048):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        if type(max_length) is not int or not 1 <= max_length <= 32768:
            raise ValueError("Max length must be between 1 and 32768 tokens")
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        # Heavy/network-capable dependency is optional and loaded only on an explicit run.
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install generate_embedding/requirements-gemini.txt first") from exc
        self._types = types
        self.client = genai.Client(api_key=api_key, http_options={
            "timeout": 45000, "retry_options": {"attempts": 1}})
        self.batch_size = batch_size
        self.max_length = max_length

    def _encode(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be nonempty text")
        # No local tokenizer for a hosted model; a conservative chars-per-token
        # proxy catches a runaway input before spending an API call on it.
        for index, text in enumerate(texts):
            if len(text) > self.max_length * 4:
                raise ValueError(f"Input {index + 1} exceeds --max-length {self.max_length} "
                                 f"(approx., by character count): {text[:80]!r}...")
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self.client.models.embed_content(
                model=EMBEDDING_MODEL, contents=batch,
                config=self._types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
            )
            for embedding in response.embeddings:
                values = list(embedding.values)
                # Google's docs: only the native (untruncated) output is
                # pre-normalized. Any other output_dimensionality -- which is
                # exactly what EMBEDDING_DIMENSIONS uses -- comes back
                # un-normalized and must be normalized client-side.
                norm = math.sqrt(sum(v * v for v in values))
                if norm == 0:
                    raise ValueError("The embedding model returned a zero vector")
                results.append([v / norm for v in values])
        validate_vectors(results, len(texts))
        return results

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, DOCUMENT_TASK_TYPE)

    def encode_query(self, query: str) -> list[float]:
        return self._encode([query], QUERY_TASK_TYPE)[0]


def write_embeddings(document: MenuEmbeddings, output: Path) -> None:
    """Replace the previous complete artifact only after the new one is ready."""
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=output.name + ".", suffix=".tmp", delete=False) as handle:
            pending = Path(handle.name)
            handle.write(document.model_dump_json())
        pending.replace(output)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def build_embeddings(input_path: Path, output_path: Path, *, batch_size: int = 10,
                     max_length: int = 2048, api_key: str | None = None) -> MenuEmbeddings:
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Embedding output must not overwrite the source menu")
    if output_path.suffix.lower() != ".json":
        raise ValueError("Embedding output must use a .json filename")
    menu, menu_sha256 = read_menu(input_path)
    started = time.monotonic()
    embedder = GeminiEmbedder(api_key=api_key, batch_size=batch_size, max_length=max_length)
    vectors = embedder.encode_documents([product.search_text for product in menu.products])
    validate_vectors(vectors, len(menu.products))
    result = MenuEmbeddings(
        menu_file=input_path.as_posix(), menu_sha256=menu_sha256,
        recipe=EmbeddingRecipe(max_length=max_length),
        created_at=datetime.now(timezone.utc).isoformat(), batch_size=batch_size,
        duration_seconds=round(time.monotonic() - started, 3), product_count=len(menu.products),
        products=[{"product_id": product.product_id,
                   "text_sha256": hashlib.sha256(product.search_text.encode("utf-8")).hexdigest(),
                   "vector": vector}
                  for product, vector in zip(menu.products, vectors, strict=True)],
    )
    # Avoid publishing vectors for an input file edited while inference was running.
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != menu_sha256:
        raise ValueError("The menu changed during embedding; rerun against the updated file")
    write_embeddings(result, output_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("structure.json"))
    parser.add_argument("--output", type=Path, default=Path("generate_embedding/data/menu-embeddings.json"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 1 <= args.max_length <= 32768:
        parser.error("Require --batch-size > 0 and 1 <= --max-length <= 32768")
    try:
        result = build_embeddings(args.input, args.output, batch_size=args.batch_size, max_length=args.max_length)
        print(f"Saved {result.product_count} product vectors ({EMBEDDING_DIMENSIONS} dimensions) to {args.output}")
        return 0
    except Exception as exc:
        print(f"Embedding failed; existing output was not replaced: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
