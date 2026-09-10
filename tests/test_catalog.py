import unittest

from app.models.menu import StructuredMenu
from app.services.build_catalog import build_catalog, digest


class CatalogTests(unittest.TestCase):
    def fixture(self):
        def source(number, description, price, section=None):
            return dict(source_id=f"source-{number:02d}", ocr_file="test.json",
                ocr_sha256="test", image_file="test.png", image_sha256="test",
                ocr_uncertainties=[], transcription="latte", menu=dict(items=[dict(
                    name="latte", category=None, section=section, description=description,
                    variants=[dict(size="S", temperature="iced", price=price, price_note=None)],
                    notes=[])], addons=[], notes=[]))
        document = StructuredMenu(model="test", model_digest="test", currency=None,
            currency_source="unspecified", source_count=2, item_occurrence_count=2,
            sources=[source(1, "Coffee with cream", 18.0), source(2, None, 18.0)])
        review = dict(input_digest=digest(document.model_dump()), separate_sections=["house blend", "seasonal specials"],
                      series=[], items={})
        return document, review

    def test_merge_preserves_description_price_and_both_sources(self):
        document, review = self.fixture()
        result = build_catalog(document, review)
        self.assertEqual(result["product_count"], 1)
        product = result["products"][0]
        self.assertEqual(product["descriptions"][0]["text"], "Coffee with cream")
        self.assertEqual(len(product["variants"]), 1)
        self.assertEqual(len(product["variants"][0]["source_refs"]), 2)

    def test_blends_do_not_merge_and_conflicting_prices_are_preserved(self):
        document, review = self.fixture()
        document.sources[1].menu.items[0].variants[0].price = 20.0
        review["input_digest"] = digest(document.model_dump())
        product = build_catalog(document, review)["products"][0]
        self.assertEqual(len(product["variants"]), 2)
        self.assertTrue(product["issues"])
        document.sources[0].menu.items[0].section = "house blend"
        document.sources[1].menu.items[0].section = "seasonal specials"
        review["input_digest"] = digest(document.model_dump())
        self.assertEqual(build_catalog(document, review)["product_count"], 2)

    def test_changed_extraction_cannot_reuse_review(self):
        document, review = self.fixture()
        document.sources.reverse()
        with self.assertRaisesRegex(ValueError, "Extraction changed"):
            build_catalog(document, review)

    def test_review_changes_do_not_mutate_raw_extraction(self):
        document, review = self.fixture()
        review["items"]["source-01:0"] = {"exclude": True}
        result = build_catalog(document, review)
        self.assertEqual(len(result["products"][0]["source_refs"]), 1)
        self.assertEqual(len(document.sources[0].menu.items), 1)
