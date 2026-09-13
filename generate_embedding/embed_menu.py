"""Embed a generated menu with Qwen on the model PC; no model loads on import."""
import argparse
import hashlib
import math
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from generate_embedding.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    QUERY_PROMPT,
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


class QwenEmbedder:
    """One shared model and recipe for both menu documents and future queries."""

    def __init__(self, *, device: str = "cuda", batch_size: int = 2,
                 max_length: int = 512, local_files_only: bool = False):
        if device not in ("cuda", "cpu"):
            raise ValueError("Device must be cuda or cpu")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        if type(max_length) is not int or not 1 <= max_length <= 32768:
            raise ValueError("Max length must be between 1 and 32768 tokens")
        # Heavy dependencies are optional and are loaded only by an explicit run.
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the embedding dependencies on the model PC; see generate_embedding/embeddings.md") from exc
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Run on the GPU PC with a compatible PyTorch build. "
                                   "CPU inference requires an explicit --device cpu.")
            try:
                # Check kernel compatibility before any Hugging Face download.
                torch.ones(1, device="cuda").add_(1)
                torch.cuda.synchronize()
            except RuntimeError as exc:
                raise RuntimeError("CUDA kernels are incompatible with this GPU. "
                                   "GTX 1070 Ti needs a Pascal-compatible build; see generate_embedding/embeddings.md") from exc
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install generate_embedding/requirements-embeddings.txt on the model PC first") from exc
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            device=device,
            trust_remote_code=False,
            local_files_only=local_files_only,
            model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager", "use_safetensors": True},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = max_length
        self.model.eval()
        if self.model.get_sentence_embedding_dimension() != EMBEDDING_DIMENSIONS:
            raise ValueError("Unexpected embedding dimension for the pinned Qwen model")

    def _encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be nonempty text")
        # Check complete text, instruction and special tokens before encode can truncate.
        for start in range(0, len(texts), self.batch_size):
            lengths = self.model.tokenizer(
                [prompt + text for text in texts[start:start + self.batch_size]],
                padding=False, truncation=False, add_special_tokens=True, return_length=True,
            )["length"]
            for offset, length in enumerate(lengths):
                if length > self.max_length:
                    raise ValueError(f"Input {start + offset + 1} needs {length} tokens, exceeding "
                                     f"--max-length {self.max_length}. Text was not truncated; "
                                     "review search_text or increase the limit on a capable PC.")
        result = self.model.encode(
            texts, prompt=prompt, batch_size=self.batch_size, show_progress_bar=False,
            normalize_embeddings=True, precision="float32", convert_to_numpy=True,
        ).tolist()
        validate_vectors(result, len(texts))
        return result

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        # Qwen documents have no instruction. An explicit empty prompt overrides defaults.
        return self._encode(texts, prompt="")

    def encode_query(self, query: str) -> list[float]:
        return self._encode([query], prompt=QUERY_PROMPT)[0]


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


def build_embeddings(input_path: Path, output_path: Path, *, batch_size: int = 2,
                     max_length: int = 512, device: str = "cuda",
                     local_files_only: bool = False) -> MenuEmbeddings:
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Embedding output must not overwrite the source menu")
    if output_path.suffix.lower() != ".json":
        raise ValueError("Embedding output must use a .json filename")
    menu, menu_sha256 = read_menu(input_path)
    started = time.monotonic()
    embedder = QwenEmbedder(device=device, batch_size=batch_size, max_length=max_length,
                            local_files_only=local_files_only)
    vectors = embedder.encode_documents([product.search_text for product in menu.products])
    validate_vectors(vectors, len(menu.products))
    result = MenuEmbeddings(
        menu_file=input_path.as_posix(), menu_sha256=menu_sha256,
        recipe=EmbeddingRecipe(max_length=max_length),
        created_at=datetime.now(timezone.utc).isoformat(), device=device, batch_size=batch_size,
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
    parser.add_argument("--output", type=Path, default=Path("data/menu-embeddings.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true", help="Use the cached model without downloading")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 1 <= args.max_length <= 32768:
        parser.error("Require --batch-size > 0 and 1 <= --max-length <= 32768")
    try:
        result = build_embeddings(args.input, args.output, batch_size=args.batch_size,
                                  max_length=args.max_length, device=args.device,
                                  local_files_only=args.local_files_only)
        print(f"Saved {result.product_count} product vectors ({EMBEDDING_DIMENSIONS} dimensions) to {args.output}")
        return 0
    except Exception as exc:
        print(f"Embedding failed; existing output was not replaced: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
