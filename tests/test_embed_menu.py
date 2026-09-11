"""Mocked embedding checks, prepared for later execution; never load real models."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.models.embeddings import EMBEDDING_DIMENSIONS, QUERY_PROMPT, MenuEmbeddings
from app.services.embed_menu import QwenEmbedder, build_embeddings, read_menu, validate_vectors


def unit_vector(index=0):
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


def menu_payload():
    products = []
    for name, description in [("Latte", "Coffee with milk"), ("Mocha", "Coffee with cocoa")]:
        products.append(dict(
            product_id=name.lower(), name=name, context=None, aliases=[], category="Coffee",
            description=description, variants=[], series=[], source_ids=["source-01"],
            evidence=[dict(source_id="source-01", quote=f"{name}: {description}")], issues=[],
            search_text=f"{name}: {description}",
        ))
    return dict(
        schema_version=3, status="draft", products=products, series=[], addons=[], issues=[],
        source_notes=[dict(source_id="source-01", notes=[])], product_count=2, source_count=1,
        currency=None, currency_source="unspecified",
        sources=[dict(source_id="source-01", ocr_file="menu.png.json", ocr_sha256="0" * 64,
                      image_file="menu.png", image_sha256="1" * 64,
                      transcription="Latte: Coffee with milk\nMocha: Coffee with cocoa", ocr_uncertainties=[])],
        generation=dict(method="qwen_batch", model="test-model", model_digest="test-digest",
                        prompt_version="test", input_digest="2" * 64, prompt_sha256="3" * 64,
                        response_sha256="4" * 64, run_id="fixture", run_directory="fixture",
                        attempt_count=1, generated_at="2026-09-11T00:00:00Z", duration_seconds=1.0,
                        num_ctx=32768, num_predict=12288, metrics={}),
    )


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.input = self.folder / "structure.json"
        self.output = self.folder / "vectors.json"
        self.input.write_text(json.dumps(menu_payload()), encoding="utf-8")

    @staticmethod
    def fake_model_embedder():
        embedder = object.__new__(QwenEmbedder)
        embedder.batch_size = 2
        embedder.max_length = 512
        embedder.device = "cpu"
        embedder.model = Mock()
        embedder.model.tokenizer.return_value = {"length": [20]}
        embedder.model.encode.return_value.tolist.return_value = [unit_vector()]
        return embedder

    def test_one_vector_per_product_without_copying_menu_facts(self):
        original = self.input.read_bytes()
        expected = [unit_vector(), unit_vector(1)]
        with patch("app.services.embed_menu.QwenEmbedder") as constructor:
            constructor.return_value.encode_documents.return_value = expected
            result = build_embeddings(self.input, self.output)
        constructor.return_value.encode_documents.assert_called_once_with(
            [p["search_text"] for p in menu_payload()["products"]])
        self.assertEqual(self.input.read_bytes(), original)
        self.assertEqual(result.menu_sha256, hashlib.sha256(original).hexdigest())
        saved = MenuEmbeddings.model_validate_json(self.output.read_bytes())
        self.assertEqual(saved.product_count, 2)
        for row, product, vector in zip(saved.products, menu_payload()["products"], expected):
            self.assertEqual(row.product_id, product["product_id"])
            self.assertEqual(row.vector, vector)
            self.assertEqual(row.text_sha256, hashlib.sha256(product["search_text"].encode()).hexdigest())
            self.assertEqual(set(row.model_dump()), {"product_id", "text_sha256", "vector"})
        self.assertEqual(saved.recipe.query_prompt, QUERY_PROMPT)

    def test_bad_menu_is_rejected_before_model_construction(self):
        bad_menus = []
        for field in ("product_count", "source_count"):
            payload = menu_payload()
            payload[field] += 1
            bad_menus.append(payload)
        payload = menu_payload()
        payload.update(products=[], product_count=0)
        bad_menus.append(payload)
        payload = menu_payload()
        payload["products"][1]["product_id"] = payload["products"][0]["product_id"]
        bad_menus.append(payload)
        payload = menu_payload()
        payload["products"][1] = copy.deepcopy(payload["products"][0])
        payload["products"][1]["product_id"] = "another-id"
        bad_menus.append(payload)
        payload = menu_payload()
        payload["products"][0]["search_text"] = " "
        bad_menus.append(payload)
        payload = menu_payload()
        payload["products"][0]["evidence"][0]["quote"] = "Invented ingredients"
        bad_menus.append(payload)
        for payload in bad_menus:
            with self.subTest(payload=payload), patch("app.services.embed_menu.QwenEmbedder") as constructor:
                self.input.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_embeddings(self.input, self.output)
                constructor.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_corrupt_or_old_menu_json_is_rejected(self):
        for raw in ("{", '{"schema_version": 2, "pages": []}'):
            with self.subTest(raw=raw):
                self.input.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_menu(self.input)

    def test_output_cannot_replace_source_menu(self):
        original = self.input.read_bytes()
        with patch("app.services.embed_menu.QwenEmbedder") as constructor:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                build_embeddings(self.input, self.input)
            constructor.assert_not_called()
        self.assertEqual(self.input.read_bytes(), original)

    def test_invalid_vectors_are_rejected(self):
        bad = [[], [unit_vector()[:-1]], [[0.0] * EMBEDDING_DIMENSIONS],
               [[2.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)]]
        for value in (float("nan"), float("inf"), True, "1"):
            vector = unit_vector()
            vector[0] = value
            bad.append([vector])
        for vectors in bad:
            with self.subTest(vectors=str(vectors)[:60]):
                with self.assertRaises(ValueError):
                    validate_vectors(vectors, 1)

    def test_failed_or_partial_inference_preserves_existing_artifact(self):
        self.output.write_bytes(b"previous complete artifact")
        for response in (RuntimeError("CUDA out of memory"), [unit_vector()]):
            with self.subTest(response=type(response).__name__), \
                 patch("app.services.embed_menu.QwenEmbedder") as constructor:
                encoder = constructor.return_value.encode_documents
                if isinstance(response, Exception):
                    encoder.side_effect = response
                else:
                    encoder.return_value = response
                with self.assertRaises((RuntimeError, ValueError)):
                    build_embeddings(self.input, self.output)
                self.assertEqual(self.output.read_bytes(), b"previous complete artifact")

    def test_changed_menu_during_inference_is_not_published(self):
        self.output.write_bytes(b"previous complete artifact")

        def change_menu(_texts):
            self.input.write_bytes(self.input.read_bytes() + b"\n")
            return [unit_vector(), unit_vector(1)]

        with patch("app.services.embed_menu.QwenEmbedder") as constructor:
            constructor.return_value.encode_documents.side_effect = change_menu
            with self.assertRaisesRegex(ValueError, "changed"):
                build_embeddings(self.input, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous complete artifact")

    def test_publish_failure_cleans_temporary_file(self):
        self.output.write_bytes(b"previous complete artifact")
        with patch("app.services.embed_menu.QwenEmbedder") as constructor, \
             patch.object(Path, "replace", side_effect=OSError("file locked")):
            constructor.return_value.encode_documents.return_value = [unit_vector(), unit_vector(1)]
            with self.assertRaises(OSError):
                build_embeddings(self.input, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous complete artifact")
        self.assertEqual(list(self.folder.glob("*.tmp")), [])

    def test_documents_and_queries_use_their_correct_prompt(self):
        embedder = self.fake_model_embedder()
        embedder.encode_documents(["Latte with milk"])
        self.assertEqual(embedder.model.encode.call_args.kwargs["prompt"], "")
        vector = embedder.encode_query("Something creamy")
        self.assertEqual(vector, unit_vector())
        self.assertEqual(embedder.model.encode.call_args.kwargs["prompt"], QUERY_PROMPT)
        self.assertTrue(embedder.model.encode.call_args.kwargs["normalize_embeddings"])
        self.assertEqual(embedder.model.tokenizer.call_args.args[0], [QUERY_PROMPT + "Something creamy"])

    def test_overlong_input_is_rejected_before_any_embedding(self):
        embedder = self.fake_model_embedder()
        embedder.model.tokenizer.side_effect = [{"length": [10, 20]}, {"length": [513]}]
        with self.assertRaisesRegex(ValueError, "not truncated"):
            embedder.encode_documents(["First", "Second", "Third"])
        embedder.model.encode.assert_not_called()

    def test_empty_query_is_rejected(self):
        embedder = self.fake_model_embedder()
        with self.assertRaises(ValueError):
            embedder.encode_query("  ")
        embedder.model.encode.assert_not_called()

    def test_missing_cuda_does_not_download_or_fall_back_to_cpu(self):
        cuda = Mock()
        cuda.is_available.return_value = False
        constructor = Mock()
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(cuda=cuda),
            "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor),
        }):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                QwenEmbedder()
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
