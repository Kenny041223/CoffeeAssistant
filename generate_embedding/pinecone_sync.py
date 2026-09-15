"""Sync generate_embedding/data/menu-embeddings.json vectors to a Pinecone index; no model loads
on import and no network calls happen under --dry-run.

Setup only, by design: this module is written to be inspected and dry-run
before it ever touches a real Pinecone account. It requires PINECONE_API_KEY
in the environment for any non-dry-run operation and never accepts the key as
a CLI argument (so it never lands in shell history or a process list).

Each vector's metadata is flattened for Pinecone's restriction to strings,
numbers, booleans, and lists of strings -- structured price/variant data
becomes parallel lists and a human-readable summary string, not nested
objects. A product's full price/variant/evidence detail always stays in
structure.json; Pinecone only gets what's useful for filtering and display
alongside a similarity match.

Sync is two-way: product_ids present in the index but no longer in the
current menu are deleted, not just left stale (matching the design intent
noted in generate_embedding/embeddings.md -- "Future database imports must also remove
products that disappeared from the latest menu").
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from generate_embedding.embeddings import EMBEDDING_DIMENSIONS, MenuEmbeddings
from generate_embedding.menu import StructuredMenu

DEFAULT_INDEX = "coffee-menu"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
DEFAULT_METRIC = "cosine"


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def load_embeddings(path: Path) -> MenuEmbeddings:
    return MenuEmbeddings.model_validate_json(path.read_text(encoding="utf-8"))


def load_menu(path: Path) -> tuple[StructuredMenu, str]:
    raw = path.read_bytes()
    return StructuredMenu.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def check_consistency(embeddings: MenuEmbeddings, menu: StructuredMenu, menu_sha256: str) -> None:
    """Refuse to sync vectors that don't demonstrably match the current menu."""
    if embeddings.menu_sha256 != menu_sha256:
        raise ValueError(
            "menu-embeddings.json was generated from a different structure.json "
            "(menu_sha256 mismatch). Regenerate embeddings against the current menu first."
        )
    by_id = {p.product_id for p in menu.products}
    text_by_id = {p.product_id: p.search_text for p in menu.products}
    for record in embeddings.products:
        if record.product_id not in by_id:
            raise ValueError(
                f"Embedding for {record.product_id} has no matching product in the current "
                "menu; regenerate embeddings so stale product IDs aren't pushed."
            )
        actual = hashlib.sha256(text_by_id[record.product_id].encode("utf-8")).hexdigest()
        if actual != record.text_sha256:
            raise ValueError(
                f"search_text for {record.product_id} changed since embeddings were generated "
                "(text_sha256 mismatch); regenerate embeddings before syncing."
            )


def build_metadata(product) -> dict:
    """Flatten one product into Pinecone-legal metadata (str/num/bool/list[str] only)."""
    sizes, temperatures, prices, price_lines = [], [], [], []
    for variant in product.variants:
        if variant.size:
            sizes.append(variant.size)
        if variant.temperature:
            temperatures.append(variant.temperature)
        if variant.price is not None:
            prices.append(variant.price)
            price_lines.append(f"{variant.size or '?'}/{variant.temperature or '?'}={variant.price}")
    metadata = {
        "product_id": product.product_id,
        "name": product.name,
        "category": product.category or "",
        "context": product.context or "",
        "description": product.description or "",
        "aliases": list(product.aliases),
        "series": list(product.series),
        "sizes": sorted(set(sizes)),
        "temperatures": sorted(set(temperatures)),
        "price_summary": ", ".join(price_lines) if price_lines else "no priced variants",
    }
    if prices:
        metadata["price_min"] = min(prices)
        metadata["price_max"] = max(prices)
    return metadata


def existing_vector_ids(index, namespace: str | None) -> set[str]:
    """index.list() yields ListResponse pages, each holding .vectors: a list of
    ListItem(id=...) -- not plain ID strings. Isolated here since that exact
    shape assumption is what broke the first time this ran against a
    non-empty index (every earlier sync only ever saw a freshly-created,
    empty one, so the bug was latent and untested)."""
    ids: set[str] = set()
    for page in index.list(namespace=namespace):
        ids.update(item.id for item in page.vectors)
    return ids


def plan_sync(embeddings: MenuEmbeddings, menu: StructuredMenu, existing_ids: set[str]) -> tuple[list[dict], set[str]]:
    """Pure computation: what to upsert and what to delete. No network calls."""
    products_by_id = {p.product_id: p for p in menu.products}
    upserts = [
        {"id": record.product_id, "values": record.vector,
         "metadata": build_metadata(products_by_id[record.product_id])}
        for record in embeddings.products
    ]
    current_ids = {record.product_id for record in embeddings.products}
    to_delete = existing_ids - current_ids
    return upserts, to_delete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=Path("generate_embedding/data/menu-embeddings.json"))
    parser.add_argument("--menu", type=Path, default=Path("structure.json"))
    parser.add_argument("--index", default=os.environ.get("PINECONE_INDEX", DEFAULT_INDEX))
    parser.add_argument("--namespace", default=os.environ.get("PINECONE_NAMESPACE", ""))
    parser.add_argument("--cloud", default=os.environ.get("PINECONE_CLOUD", DEFAULT_CLOUD))
    parser.add_argument("--region", default=os.environ.get("PINECONE_REGION", DEFAULT_REGION))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute and print the sync plan; make zero Pinecone network calls.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    try:
        if not args.embeddings.is_file():
            raise ValueError(f"Embeddings file not found: {args.embeddings}. "
                              "Run generate_embedding/generate_embeddings.ps1 first.")
        if not args.menu.is_file():
            raise ValueError(f"Menu file not found: {args.menu}. "
                              "Run generate_embedding/generate_structureFile.ps1 first.")
        embeddings = load_embeddings(args.embeddings)
        menu, menu_sha256 = load_menu(args.menu)
        check_consistency(embeddings, menu, menu_sha256)

        if args.dry_run:
            upserts, to_delete = plan_sync(embeddings, menu, existing_ids=set())
            print(f"[dry-run] No network calls made. Index={args.index!r} namespace={args.namespace!r} "
                  f"cloud={args.cloud} region={args.region} metric={DEFAULT_METRIC} dim={EMBEDDING_DIMENSIONS}")
            print(f"[dry-run] Would upsert {len(upserts)} vector(s):")
            for record in upserts:
                print(f"  - {record['id']}: {record['metadata']['name']!r} "
                      f"[{record['metadata']['category'] or 'uncategorized'}] "
                      f"{record['metadata']['price_summary']}")
            print("[dry-run] Cannot compute deletions without querying the live index "
                  "(skipped in dry-run); run without --dry-run to see and apply them.")
            return 0

        # Heavy/network-capable dependency loaded only for a real run.
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise RuntimeError(
                "Install the pinecone package first: "
                "pip install -r generate_embedding/requirements-pinecone.txt"
            ) from exc
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set. Set it in your shell "
                               "environment first; this tool never accepts it as an argument.")

        client = Pinecone(api_key=api_key)
        if not client.has_index(args.index):
            print(f"Index {args.index!r} does not exist; creating it "
                  f"(dimension={EMBEDDING_DIMENSIONS}, metric={DEFAULT_METRIC}, "
                  f"serverless {args.cloud}/{args.region}).")
            client.create_index(name=args.index, dimension=EMBEDDING_DIMENSIONS, metric=DEFAULT_METRIC,
                                 spec=ServerlessSpec(cloud=args.cloud, region=args.region))
        index = client.Index(args.index)

        existing_ids = existing_vector_ids(index, args.namespace or None)

        upserts, to_delete = plan_sync(embeddings, menu, existing_ids)
        for start in range(0, len(upserts), args.batch_size):
            index.upsert(vectors=upserts[start:start + args.batch_size], namespace=args.namespace or None)
        if to_delete:
            index.delete(ids=sorted(to_delete), namespace=args.namespace or None)

        print(f"Synced {len(upserts)} product vector(s) to index {args.index!r} "
              f"(namespace={args.namespace or '(default)'}); removed {len(to_delete)} stale vector(s).")
        return 0
    except Exception as exc:
        print(f"Pinecone sync failed; index was not fully updated: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
