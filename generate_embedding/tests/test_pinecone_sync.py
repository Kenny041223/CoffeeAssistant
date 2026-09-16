"""Mocked Pinecone sync checks; never call the real API."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from generate_embedding.menu import StructuredMenu
from generate_embedding.pinecone_sync import (
    build_metadata, check_consistency, existing_vector_ids, load_embeddings, load_menu, plan_sync,
)


def menu_payload():
    products = [
        dict(
            product_id="latte-permanent", name="Latte", context="house blend", aliases=["white"],
            category="coffee", availability="permanent", description="Nutty and strong.",
            variants=[dict(size="S", temperature="hot", price=12.0, price_note=None, source_ids=["source-01"]),
                      dict(size="L", temperature="hot", price=15.0, price_note=None, source_ids=["source-01"])],
            series=[], source_ids=["source-01"],
            evidence=[dict(source_id="source-01", quote="Latte")], issues=[],
            search_text="latte house blend",
        ),
        dict(
            product_id="tiramisu", name="Tiramisu", context=None, aliases=[], category="dessert",
            description=None, variants=[], series=[], source_ids=["source-02"],
            evidence=[dict(source_id="source-02", quote="Tiramisu")], issues=[],
            search_text="tiramisu dessert",
        ),
    ]
    return dict(
        schema_version=5, status="draft", products=products, series=[], addons=[], issues=[],
        source_notes=[dict(source_id="source-01", notes=[]), dict(source_id="source-02", notes=[])],
        product_count=2, source_count=2, currency=None, currency_source="unspecified",
        sources=[dict(source_id="source-01", ocr_file="a.json", ocr_sha256="0" * 64,
                      image_file="a.png", image_sha256="1" * 64, transcription="Latte", ocr_uncertainties=[]),
                 dict(source_id="source-02", ocr_file="b.json", ocr_sha256="2" * 64,
                      image_file="b.png", image_sha256="3" * 64, transcription="Tiramisu", ocr_uncertainties=[])],
        generation=dict(method="qwen_staged", model="test-model", model_digest="test-digest",
                        prompt_version="test", input_digest="4" * 64, prompt_sha256="5" * 64,
                        response_sha256="6" * 64, run_id="fixture", run_directory="fixture",
                        attempt_count=1, generated_at="2026-09-11T00:00:00Z", duration_seconds=1.0,
                        num_ctx=32768, num_predict=12288, metrics={}),
    )


def unit_vector(index=0, dim=768):
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def embeddings_payload(menu):
    return dict(
        schema_version=2, menu_file="structure.json",
        menu_sha256=hashlib.sha256(json.dumps(menu).encode("utf-8")).hexdigest(),
        recipe=dict(model="gemini-embedding-001", dimensions=768, normalized=True, similarity="cosine",
                    document_task_type="RETRIEVAL_DOCUMENT", query_task_type="RETRIEVAL_QUERY", max_length=2048),
        created_at="2026-09-11T00:00:00Z", provider="gemini", batch_size=10, duration_seconds=1.0,
        product_count=2,
        products=[
            {"product_id": "latte-permanent",
             "text_sha256": hashlib.sha256(b"latte house blend").hexdigest(), "vector": unit_vector(0)},
            {"product_id": "tiramisu",
             "text_sha256": hashlib.sha256(b"tiramisu dessert").hexdigest(), "vector": unit_vector(1)},
        ],
    )


class ExistingVectorIdsTests(unittest.TestCase):
    def test_extracts_ids_from_listresponse_pages_not_the_page_object_itself(self):
        # Matches the real pinecone SDK shape: index.list() yields page objects
        # with a .vectors list of ListItem(id=...), not plain ID strings or
        # directly-iterable/hashable pages. This is exactly the assumption
        # that broke on first contact with a non-empty index.
        page1 = SimpleNamespace(vectors=[SimpleNamespace(id="a"), SimpleNamespace(id="b")])
        page2 = SimpleNamespace(vectors=[SimpleNamespace(id="c")])
        index = Mock()
        index.list.return_value = [page1, page2]
        self.assertEqual(existing_vector_ids(index, None), {"a", "b", "c"})
        index.list.assert_called_once_with(namespace=None)

    def test_empty_index_returns_empty_set(self):
        index = Mock()
        index.list.return_value = []
        self.assertEqual(existing_vector_ids(index, "staging"), set())
        index.list.assert_called_once_with(namespace="staging")


class BuildMetadataTests(unittest.TestCase):
    def test_flattens_variants_into_pinecone_legal_types(self):
        menu = StructuredMenu.model_validate(menu_payload())
        metadata = build_metadata(menu.products[0])
        self.assertEqual(metadata["product_id"], "latte-permanent")
        self.assertEqual(metadata["sizes"], ["L", "S"])
        self.assertEqual(metadata["temperatures"], ["hot"])
        self.assertEqual(metadata["price_min"], 12.0)
        self.assertEqual(metadata["price_max"], 15.0)
        self.assertIn("S/hot=12.0", metadata["price_summary"])
        for value in metadata.values():
            self.assertIsInstance(value, (str, int, float, bool, list))
            if isinstance(value, list):
                for item in value:
                    self.assertIsInstance(item, str)

    def test_product_with_no_priced_variants_has_no_price_min_max(self):
        menu = StructuredMenu.model_validate(menu_payload())
        metadata = build_metadata(menu.products[1])
        self.assertNotIn("price_min", metadata)
        self.assertNotIn("price_max", metadata)
        self.assertEqual(metadata["price_summary"], "no priced variants")


class PlanSyncTests(unittest.TestCase):
    def test_upserts_every_embedded_product_and_deletes_only_ids_no_longer_present(self):
        menu = StructuredMenu.model_validate(menu_payload())
        from generate_embedding.embeddings import MenuEmbeddings
        embeddings = MenuEmbeddings.model_validate(embeddings_payload(menu_payload()))
        existing_ids = {"latte-permanent", "tiramisu", "removed-product"}

        upserts, to_delete = plan_sync(embeddings, menu, existing_ids)

        self.assertEqual({u["id"] for u in upserts}, {"latte-permanent", "tiramisu"})
        self.assertEqual(to_delete, {"removed-product"})
        for record in upserts:
            self.assertEqual(len(record["values"]), 768)
            self.assertEqual(record["metadata"]["product_id"], record["id"])

    def test_nothing_to_delete_when_index_exactly_matches_menu(self):
        menu = StructuredMenu.model_validate(menu_payload())
        from generate_embedding.embeddings import MenuEmbeddings
        embeddings = MenuEmbeddings.model_validate(embeddings_payload(menu_payload()))
        _, to_delete = plan_sync(embeddings, menu, {"latte-permanent", "tiramisu"})
        self.assertEqual(to_delete, set())


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)

    def test_mismatched_menu_hash_is_rejected(self):
        from generate_embedding.embeddings import MenuEmbeddings
        menu = StructuredMenu.model_validate(menu_payload())
        payload = embeddings_payload(menu_payload())
        payload["menu_sha256"] = "0" * 64
        embeddings = MenuEmbeddings.model_validate(payload)
        with self.assertRaisesRegex(ValueError, "menu_sha256 mismatch"):
            check_consistency(embeddings, menu, hashlib.sha256(b"current").hexdigest())

    def test_product_id_no_longer_in_menu_is_rejected(self):
        from generate_embedding.embeddings import MenuEmbeddings
        menu_raw = menu_payload()
        current_sha = hashlib.sha256(json.dumps(menu_raw).encode("utf-8")).hexdigest()
        payload = embeddings_payload(menu_raw)
        payload["products"].append(
            {"product_id": "gone", "text_sha256": "a" * 64, "vector": unit_vector(2)})
        payload["product_count"] = 3
        embeddings = MenuEmbeddings.model_validate(payload)
        menu = StructuredMenu.model_validate(menu_raw)
        with self.assertRaisesRegex(ValueError, "no matching product"):
            check_consistency(embeddings, menu, current_sha)

    def test_changed_search_text_is_rejected(self):
        from generate_embedding.embeddings import MenuEmbeddings
        menu_raw = menu_payload()
        current_sha = hashlib.sha256(json.dumps(menu_raw).encode("utf-8")).hexdigest()
        payload = embeddings_payload(menu_raw)
        payload["products"][0]["text_sha256"] = "f" * 64
        embeddings = MenuEmbeddings.model_validate(payload)
        menu = StructuredMenu.model_validate(menu_raw)
        with self.assertRaisesRegex(ValueError, "text_sha256 mismatch"):
            check_consistency(embeddings, menu, current_sha)

    def test_matching_menu_and_embeddings_pass(self):
        from generate_embedding.embeddings import MenuEmbeddings
        menu_raw = menu_payload()
        current_sha = hashlib.sha256(json.dumps(menu_raw).encode("utf-8")).hexdigest()
        embeddings = MenuEmbeddings.model_validate(embeddings_payload(menu_raw))
        menu = StructuredMenu.model_validate(menu_raw)
        check_consistency(embeddings, menu, current_sha)  # must not raise

    def test_incomplete_embeddings_cannot_schedule_current_products_for_deletion(self):
        from generate_embedding.embeddings import MenuEmbeddings
        raw = menu_payload()
        menu = StructuredMenu.model_validate(raw)
        payload = embeddings_payload(raw)
        payload["products"] = payload["products"][:1]
        for count in (1, 2):
            payload["product_count"] = count
            embeddings = MenuEmbeddings.model_validate(payload)
            with self.assertRaisesRegex(ValueError, "cover every"):
                check_consistency(embeddings, menu, payload["menu_sha256"])
            with self.assertRaisesRegex(ValueError, "cover every"):
                plan_sync(embeddings, menu, {p.product_id for p in menu.products})

    def test_duplicate_ids_wrong_counts_and_bad_vectors_are_rejected(self):
        from generate_embedding.embeddings import MenuEmbeddings
        for defect in ("duplicate_embedding", "duplicate_menu", "embedding_count", "menu_count", "zero_vector"):
            with self.subTest(defect=defect):
                raw = menu_payload()
                payload = embeddings_payload(raw)
                if defect == "duplicate_embedding":
                    payload["products"][1] = payload["products"][0]
                elif defect == "duplicate_menu":
                    raw["products"][1] = raw["products"][0]
                elif defect == "embedding_count":
                    payload["product_count"] = 99
                elif defect == "menu_count":
                    raw["product_count"] = 99
                else:
                    payload["products"][0]["vector"] = [0.0] * 768
                menu = StructuredMenu.model_validate(raw)
                embeddings = MenuEmbeddings.model_validate(payload)
                with self.assertRaises(ValueError):
                    plan_sync(embeddings, menu, {"latte-permanent", "tiramisu"})


if __name__ == "__main__":
    unittest.main()
