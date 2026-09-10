import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
from pydantic import ValidationError
from app.models.qwen_ocr import QwenReading, QwenDocument
from app.services.ocr import QwenEngine, extract_image, find_images, write_result


class OCRTests(unittest.TestCase):
    def test_roundtrip_without_fabricated_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "menu.png"
            Image.new("RGB", (80, 40), "red").save(source)
            engine = Mock(return_value=QwenReading(transcription="| hot | iced |\n| 16.0 | |", uncertainties=[]))
            engine.model = "qwen3-vl:4b-instruct"
            engine.model_digest = "test-digest"
            doc = extract_image(engine, source)
            self.assertEqual(engine.call_args.args[0].mode, "RGB")
            self.assertFalse(doc.verified)
            self.assertNotIn("confidence", doc.model_dump())
            write_result(doc, Path(directory), source.name)
            self.assertEqual(QwenDocument.model_validate_json(
                (Path(directory) / "menu.png.json").read_bytes()), doc)

    def test_empty_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                find_images(Path(directory))

    def test_blank_transcription_remains_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "blank.png"
            Image.new("RGB", (20, 20)).save(source)
            engine = Mock(return_value=QwenReading(transcription="", uncertainties=[]))
            engine.model, engine.model_digest = "qwen", "test"
            document = extract_image(engine, source)
            self.assertEqual(document.status, "no_text")
            self.assertFalse(document.verified)

    def test_invented_fields_rejected(self):
        with self.assertRaises(ValidationError):
            QwenReading(transcription="16.0", uncertainties=[], confidence=0.99)

    def test_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            QwenEngine("https://example.com", "qwen", 60)

    @patch("app.services.ocr.requests.Session")
    def test_api_image_payload_and_truncation(self, session):
        session.return_value.get.return_value.json.return_value = {
            "models": [{"name": "qwen", "digest": "abc"}]}
        post = session.return_value.post
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"done": True, "done_reason": "stop",
            "message": {"content": '{"transcription":"16.0","uncertainties":[]}'}}
        engine = QwenEngine("http://127.0.0.1:11435", "qwen", 60)
        self.assertEqual(engine(Image.new("RGB", (20, 20))).transcription, "16.0")
        self.assertTrue(post.call_args.kwargs["json"]["messages"][0]["images"])
        post.return_value.json.return_value["done_reason"] = "length"
        with self.assertRaises(ValueError):
            engine(Image.new("RGB", (20, 20)))
        post.return_value.json.return_value = {"done": True, "done_reason": "stop",
            "message": {"content": '{"transcription": "16.0"}'}}
        with self.assertRaises(ValidationError):
            engine(Image.new("RGB", (20, 20)))


if __name__ == "__main__":
    unittest.main()
