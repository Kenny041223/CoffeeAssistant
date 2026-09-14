"""Offline checks for the model-generated menu pipeline; never start Ollama."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from generate_embedding.menu import ModelMenu, MenuVariant
from generate_embedding.qwen_ocr import QwenDocument
from generate_embedding.structure_menu import (
    build_messages, generate_menu, main, read_sources, validate_model_menu,
)


class StructureTests(unittest.TestCase):
    @staticmethod
    def sources():
        return [
            dict(source_id="source-01", transcription="Vienna latte\nSmall iced 18.0\nLarge iced 20.0",
                 ocr_uncertainties=[]),
            dict(source_id="source-02", transcription="Vienna latte\nCoffee topped with whipped cream and cocoa.",
                 ocr_uncertainties=[]),
        ]

    @staticmethod
    def payload():
        return dict(
            products=[dict(
                name="Vienna latte", context=None, aliases=[], category=None, availability=None,
                description="Coffee topped with whipped cream and cocoa.",
                variants=[
                    dict(size="small", temperature="iced", price=18.0,
                         price_note=None, source_ids=["source-01"]),
                    dict(size="large", temperature="iced", price=20.0,
                         price_note=None, source_ids=["source-01"]),
                ],
                series=[], source_ids=["source-01", "source-02"],
                evidence=[dict(source_id="source-01", quote="Vienna latte\nSmall iced 18.0\nLarge iced 20.0"),
                          dict(source_id="source-02", quote="Coffee topped with whipped cream and cocoa.")],
                issues=[],
                search_text="Vienna latte: iced coffee with whipped cream and cocoa; small and large sizes.",
            )],
            series=[], addons=[],
            source_notes=[dict(source_id="source-01", notes=[]), dict(source_id="source-02", notes=[])],
            issues=[],
        )

    @staticmethod
    def document(text="Vienna latte", name="menu.png"):
        return QwenDocument(
            source_image=f"C:/another-pc/menu-images/{name}", source_sha256="image-hash",
            width=100, height=100, model="qwen-vision", model_digest="vision-digest",
            status="text_detected" if text.strip() else "no_text", transcription=text,
            uncertainties=[], duration_seconds=1,
        )

    def write_batch(self, directory):
        folder = Path(directory, "ocr")
        folder.mkdir()
        processed = []
        for index, source in enumerate(self.sources(), 1):
            name = f"menu-{index}.png"
            document = self.document(source["transcription"], name)
            (folder / f"{name}.json").write_text(document.model_dump_json(), encoding="utf-8")
            processed.append(dict(image=name, status=document.status))
        (folder / "summary.json").write_text(json.dumps(dict(
            model="qwen-vision", model_digest="vision-digest", processed=processed, failed=[],
        )), encoding="utf-8")
        return folder

    @staticmethod
    def engine(raw):
        engine = Mock(model="qwen-text", model_digest="text-digest", last_generation={})
        engine.generate_json.return_value = raw
        return engine

    @staticmethod
    def run_cli(args):
        with patch("sys.argv", ["structure_menu", *map(str, args)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as error:
            return main(), error.getvalue()

    def test_all_ocr_pages_share_one_request_and_json_schema(self):
        schema = ModelMenu.model_json_schema()
        messages = build_messages(self.sources(), schema)
        request_text = "\n".join(message["content"] for message in messages)
        for source in self.sources():
            self.assertIn(source["source_id"], request_text)
            self.assertIn(source["transcription"].splitlines()[-1], request_text)
        with tempfile.TemporaryDirectory() as directory:
            raw = json.dumps(self.payload())
            engine = self.engine(raw)
            menu, response, attempts = generate_menu(
                engine, self.sources(), Path(directory), 32768, 12288)
        engine.generate_json.assert_called_once()
        self.assertEqual(engine.generate_json.call_args.args[1], schema)
        self.assertEqual(menu.model_dump(), self.payload())
        self.assertEqual(response, raw)
        self.assertEqual(attempts, 1)

    def test_duplicate_product_is_repaired_by_model_instead_of_merged_in_python(self):
        duplicate = self.payload()
        duplicate["products"].append(copy.deepcopy(duplicate["products"][0]))
        first_response, repaired_response = json.dumps(duplicate), json.dumps(self.payload())
        engine = self.engine("")
        requests = []

        def answer(messages, schema):
            requests.append(copy.deepcopy(messages))
            return first_response if len(requests) == 1 else repaired_response

        engine.generate_json.side_effect = answer
        with tempfile.TemporaryDirectory() as directory:
            menu, raw, attempts = generate_menu(engine, self.sources(), Path(directory), 32768, 12288)
            artifacts = [p.read_text(encoding="utf-8") for p in Path(directory).rglob("*") if p.is_file()]
        self.assertEqual(attempts, 2)
        self.assertEqual(raw, repaired_response)
        self.assertEqual(menu.model_dump(), self.payload())
        self.assertIn(first_response, artifacts)
        self.assertIn(repaired_response, artifacts)
        feedback = "\n".join(message["content"] for message in requests[1])
        self.assertIn("duplicate", feedback.lower())
        for source in self.sources():
            self.assertIn(source["source_id"], feedback)
            self.assertIn(source["transcription"].splitlines()[-1], feedback)

    def test_two_invalid_model_responses_fail_without_an_unbounded_retry(self):
        engine = self.engine("{invalid json")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                generate_menu(engine, self.sources(), Path(directory), 32768, 12288)
            artifacts = [p.read_text(encoding="utf-8") for p in Path(directory).rglob("*") if p.is_file()]
        self.assertEqual(engine.generate_json.call_count, 2)
        self.assertGreaterEqual(artifacts.count("{invalid json"), 2)

    def test_missing_source_coverage_requests_model_repair(self):
        incomplete = self.payload()
        incomplete["source_notes"] = incomplete["source_notes"][:1]
        engine = self.engine("")
        engine.generate_json.side_effect = [json.dumps(incomplete), json.dumps(self.payload())]
        with tempfile.TemporaryDirectory() as directory:
            menu, _, attempts = generate_menu(engine, self.sources(), Path(directory), 32768, 12288)
        self.assertEqual(attempts, 2)
        self.assertEqual(len(menu.source_notes), 2)

    def test_empty_menu_from_nonempty_ocr_is_rejected(self):
        data = self.payload()
        data["products"] = []
        with self.assertRaises(ValueError):
            validate_model_menu(ModelMenu.model_validate(data), self.sources())

    def test_invalid_source_ids_and_fabricated_evidence_are_rejected(self):
        changes = {
            "product source": lambda p: p.update(source_ids=["missing"]),
            "variant source": lambda p: p["variants"][0].update(source_ids=["missing"]),
            "evidence source": lambda p: p["evidence"][0].update(source_id="missing"),
            "fabricated quote": lambda p: p["evidence"][0].update(quote="Sugar free and caffeine free"),
            "wrong quote source": lambda p: p["evidence"][1].update(source_id="source-01"),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                data = self.payload()
                change(data["products"][0])
                with self.assertRaises(ValueError):
                    validate_model_menu(ModelMenu.model_validate(data), self.sources())

    def test_same_name_in_different_blend_contexts_remains_separate(self):
        data = self.payload()
        data["products"][0]["context"] = "house blend"
        data["products"][0]["search_text"] += " House blend."
        other = copy.deepcopy(data["products"][0])
        other["context"] = "seasonal blend"
        other["search_text"] = other["search_text"].replace("House blend", "Seasonal blend")
        data["products"].append(other)
        menu = ModelMenu.model_validate(data)
        validate_model_menu(menu, self.sources())
        self.assertEqual(len(menu.products), 2)

    def test_unknown_fields_remain_null_and_description_only_product_has_no_price(self):
        data = self.payload()
        product = data["products"][0]
        product["variants"] = []
        product["category"] = None
        product["context"] = None
        menu = ModelMenu.model_validate(data)
        validate_model_menu(menu, self.sources())
        self.assertEqual(menu.model_dump(), data)
        variant = MenuVariant(size=None, temperature=None, price=None, price_note=None,
                              source_ids=["source-01"])
        self.assertIsNone(variant.price)
        self.assertIsNone(variant.size)
        self.assertIsNone(variant.temperature)

    def test_duplicate_variants_and_alias_collisions_are_rejected_without_mutation(self):
        cases = {
            "duplicate size spelling": lambda data: data["products"][0]["variants"].append(dict(
                size="S", temperature="iced", price=18.0, price_note=None, source_ids=["source-01"])),
            "canonical name as alias": lambda data: data["products"][0].update(aliases=["Vienna latte"]),
            "repeated alias": lambda data: data["products"][0].update(aliases=["Iced Vienna", "iced vienna"]),
        }
        for label, change in cases.items():
            with self.subTest(label=label):
                data = self.payload()
                change(data)
                menu = ModelMenu.model_validate(data)
                with self.assertRaises(ValueError):
                    validate_model_menu(menu, self.sources())
                self.assertEqual(menu.model_dump(), data)

    def test_negative_boolean_and_nonfinite_prices_are_rejected(self):
        for price in (-3, True, float("nan"), float("inf")):
            with self.subTest(price=price), self.assertRaises(ValidationError):
                MenuVariant(size=None, temperature=None, price=price, price_note=None,
                            source_ids=["source-01"])

    def test_schema_requires_model_search_text_and_rejects_invented_fields(self):
        for change in (lambda p: p.pop("search_text"), lambda p: p.update(confidence=0.99)):
            data = self.payload()
            change(data["products"][0])
            with self.assertRaises(ValidationError):
                ModelMenu.model_validate(data)

    def test_series_claims_and_addon_scopes_remain_separate_from_drink_variants(self):
        data, sources = self.payload(), self.sources()
        sources[0]["transcription"] += "\nCoffee series from 18.0\nOat milk +3.0 for coffee"
        data["products"][0]["series"] = ["Coffee series"]
        data["series"] = [dict(
            name="Coffee series", description=None, source_ids=["source-01"], issues=[],
            evidence=[dict(source_id="source-01", quote="Coffee series from 18.0")],
            price_claims=[dict(size=None, temperature=None, price=18.0,
                               price_note="from", source_ids=["source-01"])],
        )]
        data["addons"] = [dict(
            name="Oat milk", source_ids=["source-01"], issues=[],
            evidence=[dict(source_id="source-01", quote="Oat milk +3.0 for coffee")],
            offers=[dict(price=3.0, applies_to=["coffee"], source_ids=["source-01"])],
        )]
        menu = ModelMenu.model_validate(data)
        validate_model_menu(menu, sources)
        self.assertEqual(menu.model_dump(), data)
        self.assertEqual(len(menu.products[0].variants), 2)
        for label, change in (
            ("unknown series", lambda d: d["products"][0].update(series=["Missing group"])),
            ("unknown offer source", lambda d: d["addons"][0]["offers"][0].update(source_ids=["missing"])),
            ("unknown claim source", lambda d: d["series"][0]["price_claims"][0].update(source_ids=["missing"])),
        ):
            with self.subTest(label=label):
                invalid = copy.deepcopy(data)
                change(invalid)
                with self.assertRaises(ValueError):
                    validate_model_menu(ModelMenu.model_validate(invalid), sources)

    def test_oversized_batch_is_rejected_before_model_call(self):
        sources = self.sources()
        sources[0]["transcription"] = "Vienna latte with cream " * 30000
        engine = self.engine(json.dumps(self.payload()))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                generate_menu(engine, sources, Path(directory), 32768, 12288)
        engine.generate_json.assert_not_called()

    def test_batch_reads_copied_ocr_without_original_images_and_ignores_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.write_batch(directory)
            (folder / "stale.json").write_text("not an OCR document", encoding="utf-8")
            documents = read_sources(folder)
        self.assertEqual([path.name for path, _, _ in documents], ["menu-1.png.json", "menu-2.png.json"])
        self.assertEqual([doc.transcription for _, _, doc in documents],
                         [s["transcription"] for s in self.sources()])

    def test_failed_duplicate_missing_or_traversal_manifest_entries_are_rejected(self):
        cases = {
            "failed": lambda m: m.update(failed=[dict(image="bad.png", error="offline")]),
            "duplicate": lambda m: m["processed"].append(dict(m["processed"][0])),
            "missing document": lambda m: m["processed"][0].update(image="missing.png"),
            "parent path": lambda m: m["processed"][0].update(image="../other.png"),
            "empty batch": lambda m: m.update(processed=[]),
        }
        for label, change in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                folder = self.write_batch(directory)
                manifest = folder / "summary.json"
                data = json.loads(manifest.read_text(encoding="utf-8"))
                change(data)
                manifest.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises((ValueError, OSError)):
                    read_sources(folder)

    def test_ocr_status_must_match_transcription_and_manifest(self):
        for label in ("document status", "manifest status"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                folder = self.write_batch(directory)
                if label == "document status":
                    path = folder / "menu-1.png.json"
                    data = json.loads(path.read_text(encoding="utf-8"))
                    data["status"] = "no_text"
                else:
                    path = folder / "summary.json"
                    data = json.loads(path.read_text(encoding="utf-8"))
                    data["processed"][0]["status"] = "no_text"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_sources(folder)

    def test_prepare_only_writes_reviewable_inputs_without_model_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.write_batch(directory)
            output, run_root = Path(directory, "structure.json"), Path(directory, "runs")
            with patch("generate_embedding.structure_menu.QwenEngine") as engine:
                code, error = self.run_cli(["--input", folder, "--output", output,
                                           "--run-dir", run_root, "--prepare-only"])
            self.assertEqual(code, 0, error)
            engine.assert_not_called()
            self.assertFalse(output.exists())
            run = next(run_root.iterdir())
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "prepared")
            for name in ("inputs.json", "schema.json", "prompt.txt"):
                self.assertTrue((run / name).is_file(), name)
            inputs = (run / "inputs.json").read_text(encoding="utf-8")
            self.assertIn("Coffee topped with whipped cream and cocoa.", inputs)
            self.assertIn("source-01", inputs)
            self.assertIn("search_text", (run / "schema.json").read_text())

    def test_model_content_is_published_verbatim_with_provenance_and_no_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.write_batch(directory)
            output, run_root = Path(directory, "structure.json"), Path(directory, "runs")
            raw = json.dumps(self.payload())
            engine = self.engine(raw)
            engine.last_generation = {"prompt_eval_count": 1234, "eval_count": 567}
            with patch("generate_embedding.structure_menu.QwenEngine", return_value=engine) as factory:
                code, error = self.run_cli(["--input", folder, "--output", output,
                                           "--run-dir", run_root, "--currency", "MYR"])
            self.assertEqual(code, 0, error)
            engine.generate_json.assert_called_once()
            self.assertEqual(factory.call_args.kwargs["num_ctx"], 32768)
            self.assertEqual(factory.call_args.kwargs["num_predict"], 12288)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], 4)
            self.assertEqual(result["source_count"], 2)
            self.assertEqual(result["product_count"], 1)
            self.assertEqual(result["currency"], "MYR")
            for key, value in self.payload()["products"][0].items():
                self.assertEqual(result["products"][0][key], value, key)
            self.assertTrue(result["products"][0]["product_id"])
            self.assertEqual(result["source_notes"], self.payload()["source_notes"])
            self.assertEqual(len(result["sources"]), 2)
            for source in result["sources"]:
                self.assertNotIn("menu", source)
                self.assertNotIn("products", source)
            self.assertEqual([p.name for p in Path(directory).glob("*.json")], ["structure.json"])
            run = next(run_root.iterdir())
            self.assertEqual((run / "response-1.txt").read_text(encoding="utf-8"), raw)
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "succeeded")
            generation = result["generation"]
            self.assertEqual(generation["method"], "qwen_batch")
            self.assertEqual(generation["model_digest"], "text-digest")
            self.assertEqual(generation["response_sha256"], hashlib.sha256(raw.encode()).hexdigest())
            self.assertEqual(generation["attempt_count"], 1)
            self.assertEqual(generation["metrics"], engine.last_generation)

    def test_model_failure_or_invalid_response_preserves_existing_output_and_failed_manifest(self):
        for failure in (RuntimeError("offline"), "{}"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                folder = self.write_batch(directory)
                output, run_root = Path(directory, "structure.json"), Path(directory, "runs")
                output.write_text("previous menu", encoding="utf-8")
                engine = self.engine(failure if isinstance(failure, str) else "")
                if isinstance(failure, Exception):
                    engine.generate_json.side_effect = failure
                with patch("generate_embedding.structure_menu.QwenEngine", return_value=engine):
                    code, _ = self.run_cli(["--input", folder, "--output", output, "--run-dir", run_root])
                self.assertEqual(code, 1)
                self.assertEqual(output.read_text(encoding="utf-8"), "previous menu")
                self.assertEqual(list(Path(directory).glob("*.tmp")), [])
                run = next(run_root.iterdir())
                self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "failed")

    def test_prepare_only_keeps_existing_output_and_each_run_has_its_own_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.write_batch(directory)
            output, run_root = Path(directory, "structure.json"), Path(directory, "runs")
            output.write_text("previous menu", encoding="utf-8")
            args = ["--input", folder, "--output", output, "--run-dir", run_root, "--prepare-only"]
            with patch("generate_embedding.structure_menu.QwenEngine") as engine:
                for _ in range(2):
                    code, error = self.run_cli(args)
                    self.assertEqual(code, 0, error)
            engine.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "previous menu")
            self.assertEqual(len([p for p in run_root.iterdir() if p.is_dir()]), 2)

    def test_failed_atomic_publish_keeps_previous_output_and_cleans_pending_file(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.write_batch(directory)
            output, run_root = Path(directory, "structure.json"), Path(directory, "runs")
            output.write_text("previous menu", encoding="utf-8")
            engine = self.engine(json.dumps(self.payload()))
            with patch("generate_embedding.structure_menu.QwenEngine", return_value=engine), \
                 patch.object(Path, "replace", side_effect=OSError("permission denied")):
                code, error = self.run_cli(["--input", folder, "--output", output, "--run-dir", run_root])
            self.assertEqual(code, 1, error)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous menu")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            run = next(run_root.iterdir())
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "failed")

    def test_output_cannot_overwrite_ocr_document_or_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.write_batch(directory)
            for output in (folder / "menu-1.png.json", folder / "summary.json"):
                with self.subTest(output=output.name):
                    before = output.read_bytes()
                    with patch("generate_embedding.structure_menu.QwenEngine") as engine:
                        code, _ = self.run_cli(["--input", folder, "--output", output,
                                               "--run-dir", Path(directory, "runs")])
                    self.assertEqual(code, 1)
                    engine.assert_not_called()
                    self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
