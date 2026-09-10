import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from app.models.menu import MenuPage, MenuVariant, MenuAddon
from app.models.qwen_ocr import QwenDocument
from app.services.structure_menu import extract_page, main, read_sources


class StructureTests(unittest.TestCase):
    def document(self, text="iced latte\n18.0 (S) 20.0 (L)"):
        return QwenDocument(source_image="menu.png", source_sha256="image-hash", width=100,
                            height=100, model="qwen", model_digest="test", status="text_detected",
                            transcription=text, uncertainties=[], duration_seconds=1)

    def page(self):
        return MenuPage.model_validate({"items": [{"name": "iced latte", "category": None,
            "section": None, "description": None, "variants": [
                {"size": "S", "temperature": "iced", "price": 18.0, "price_note": None},
                {"size": "L", "temperature": "iced", "price": 20.0, "price_note": None}],
            "notes": []}], "addons": [], "notes": []})

    def test_multiple_prices_preserved(self):
        engine = Mock()
        engine.generate_json.return_value = self.page().model_dump_json()
        result = extract_page(engine, self.document())
        self.assertEqual([v.price for v in result.items[0].variants], [18.0, 20.0])
        self.assertEqual([v.size for v in result.items[0].variants], ["S", "L"])

    def test_description_only_product_can_have_no_variants(self):
        page = self.page()
        raw = page.model_dump()
        raw["items"][0]["variants"] = []
        raw["items"][0]["description"] = "Coffee topped with cream"
        parsed = MenuPage.model_validate(raw)
        self.assertEqual(parsed.items[0].variants, [])

    def test_missing_description_and_addon_scope_are_normalized(self):
        raw = self.page().model_dump()
        raw["items"][0]["description"] = ""
        self.assertIsNone(MenuPage.model_validate(raw).items[0].description)
        self.assertEqual(MenuAddon(name="oat milk", price=3, applies_to="coffee").applies_to, ["coffee"])
        self.assertEqual(MenuAddon(name="oat milk", price=3, applies_to=["coffee", "tea"]).applies_to, ["coffee", "tea"])

    def test_unknown_price_is_not_zero_and_negative_rejected(self):
        variant = MenuVariant(size=None, temperature=None, price=None, price_note=None)
        self.assertIsNone(variant.price)
        with self.assertRaises(ValidationError):
            MenuVariant(size=None, temperature=None, price=-3, price_note=None)
        with self.assertRaises(ValidationError):
            MenuVariant(size=None, temperature=None, price=True, price_note=None)

    def test_empty_extraction_from_nonempty_menu_fails(self):
        engine = Mock()
        engine.generate_json.return_value = '{"items":[],"addons":[],"notes":[]}'
        with self.assertRaises(ValueError):
            extract_page(engine, self.document())

    def test_one_retry_for_invalid_response(self):
        engine = Mock()
        engine.generate_json.side_effect = ['{}', self.page().model_dump_json()]
        extract_page(engine, self.document())
        self.assertEqual(engine.generate_json.call_count, 2)

    def test_empty_ocr_does_not_call_model(self):
        engine = Mock()
        self.assertEqual(extract_page(engine, self.document("")).items, [])
        engine.generate_json.assert_not_called()

    def test_failed_ocr_batch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "summary.json").write_text(json.dumps({"processed": [], "failed": ["bad"]}))
            with self.assertRaises(ValueError):
                read_sources(Path(directory))

    def test_model_failure_does_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "structure.json")
            output.write_text("existing")
            with patch("sys.argv", ["structure_menu", "--output", str(output)]), \
                 patch("app.services.structure_menu.read_sources", return_value=[
                     (Path(directory, "input.json"), b"{}", self.document())]), \
                 patch("app.services.structure_menu.QwenEngine", side_effect=RuntimeError("offline")):
                self.assertEqual(main(), 1)
            self.assertEqual(output.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
