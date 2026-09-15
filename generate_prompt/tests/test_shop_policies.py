"""Offline checks for shop-level policy facts; never touch the real chatbot."""
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from generate_prompt.shop_policies import ShopPolicy, format_shop_policies, load_shop_policies


class LoadShopPoliciesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)

    def test_missing_file_returns_empty_list_not_an_error(self):
        self.assertEqual(load_shop_policies(self.folder / "nope.json"), [])

    def test_valid_file_is_parsed_into_policy_objects(self):
        path = self.folder / "policies.json"
        path.write_text(json.dumps([
            {"topic": "bringing your own coffee beans",
             "answer": "Yes, for a charge; ask a barista for details."},
        ]), encoding="utf-8")
        policies = load_shop_policies(path)
        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0].topic, "bringing your own coffee beans")

    def test_non_list_json_is_rejected(self):
        path = self.folder / "bad.json"
        path.write_text(json.dumps({"topic": "x", "answer": "y"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_shop_policies(path)

    def test_invented_fields_and_blank_values_are_rejected(self):
        with self.assertRaises(ValidationError):
            ShopPolicy.model_validate({"topic": "x", "answer": "y", "confidence": 0.9})
        with self.assertRaises(ValidationError):
            ShopPolicy.model_validate({"topic": "", "answer": "y"})


class FormatShopPoliciesTests(unittest.TestCase):
    def test_empty_list_formats_as_empty_string(self):
        self.assertEqual(format_shop_policies([]), "")

    def test_policies_are_rendered_as_one_line_each(self):
        policies = [
            ShopPolicy(topic="own beans", answer="Yes, for a charge."),
            ShopPolicy(topic="wifi", answer="Free wifi, ask a barista for the password."),
        ]
        text = format_shop_policies(policies)
        self.assertIn("- own beans: Yes, for a charge.", text)
        self.assertIn("- wifi: Free wifi, ask a barista for the password.", text)
        self.assertTrue(text.startswith("Shop policies"))


if __name__ == "__main__":
    unittest.main()
