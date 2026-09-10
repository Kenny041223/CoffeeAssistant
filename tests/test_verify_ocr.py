import unittest
from unittest.mock import patch, Mock

from PIL import Image

from app.models.ocr import OCRDocument, OCRRegion
from app.services.verify_ocr import Reference, ReviewEntry, audit, compare_reference, vision_review


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.doc = OCRDocument(source_image="menu.png", source_sha256="image-hash", width=100,
                               height=100, engine="test", review_threshold=0.9, status="text_detected",
                               regions=[OCRRegion(text="16.0", confidence=0.99,
                               box=((1, 1), (40, 1), (40, 20), (1, 20)), needs_review=False)])
        self.ref = Reference(source_sha256="image-hash", ocr_sha256="ocr-hash", reviewer="Tester",
                             full_image_reviewed=True, missing_text=[],
                             entries=[ReviewEntry(region_id=0, expected_text="16.0")])

    def test_high_confidence_wrong_price_fails(self):
        self.ref.entries[0].expected_text = "18.0"
        self.assertTrue(compare_reference(self.doc, "ocr-hash", self.ref))

    def test_matching_human_reference(self):
        self.assertEqual(compare_reference(self.doc, "ocr-hash", self.ref), [])

    def test_missing_text_and_unreviewed_image_fail(self):
        self.ref.missing_text = ["iced 20.0"]
        self.ref.full_image_reviewed = False
        self.assertEqual(len(compare_reference(self.doc, "ocr-hash", self.ref)), 2)

    def test_stale_reference_fails(self):
        self.assertIn("stale", compare_reference(self.doc, "changed", self.ref)[0])

    def test_duplicate_or_pending_regions_fail(self):
        self.ref.entries.append(self.ref.entries[0])
        self.assertTrue(compare_reference(self.doc, "ocr-hash", self.ref))
        self.ref.entries = [ReviewEntry(region_id=0, expected_text=None)]
        self.assertTrue(compare_reference(self.doc, "ocr-hash", self.ref))

    def test_source_and_geometry_checks(self):
        self.doc.regions[0].box = ((-1, 1), (40, 1), (40, 20), (1, 20))
        issues = audit(self.doc, Image.new("RGB", (100, 100)), "changed")
        self.assertEqual(len(issues), 2)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"})
    @patch("requests.post")
    def test_vision_result_is_validated_and_does_not_modify_ocr(self, post):
        post.return_value = Mock(ok=True)
        post.return_value.json.return_value = {
            "status": "completed", "id": "test-response", "model": "test-model",
            "output": [{"content": [{"type": "output_text", "text":
                '{"regions":[{"region_id":0,"verdict":"mismatch","suggested_text":"18.0",'
                '"reason":"Image shows 18.0"}],"missing_text":[],"limitations":[]}'}]}]}
        result = vision_review(Image.new("RGB", (100, 100)), self.doc, "test-model")
        self.assertEqual(result["review"]["regions"][0]["verdict"], "mismatch")
        self.assertEqual(self.doc.regions[0].text, "16.0")
        self.assertFalse(post.call_args.kwargs["json"]["store"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"})
    @patch("requests.post")
    def test_incomplete_vision_response_fails(self, post):
        post.return_value = Mock(ok=True)
        post.return_value.json.return_value = {"status": "incomplete"}
        with self.assertRaises(ValueError):
            vision_review(Image.new("RGB", (100, 100)), self.doc, "test-model")


if __name__ == "__main__":
    unittest.main()
