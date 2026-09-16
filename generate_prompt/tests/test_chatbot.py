"""Chatbot regressions with real menu records and mocked API boundaries."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from generate_embedding.menu import StructuredMenu
from generate_prompt.chatbot import (
    ChatEngine, Conversation, MAX_HISTORY_CHARS, MAX_HISTORY_TURNS, SYSTEM_PROMPT,
    build_engine, format_addons, format_context, new_conversation, reply, retrieve,
)

MENU_PATH = Path(__file__).resolve().parents[2] / "structure.json"


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        self.menu = StructuredMenu.model_validate_json(MENU_PATH.read_bytes())
        self.engine = ChatEngine(
            menu_by_id={p.product_id: p for p in self.menu.products}, embedder=Mock(), index=Mock(),
            top_k=15, genai_client=Mock(), gemini_model="test-model", system_instruction=SYSTEM_PROMPT,
            addons=self.menu.addons, currency=self.menu.currency, context_version="current-menu",
        )
        self.engine.index.query.return_value = SimpleNamespace(matches=[])
        self.engine.genai_client.models.generate_content.return_value = SimpleNamespace(text="Try our latte.")

    def test_uncertainty_is_preserved_next_to_conflicting_prices(self):
        strawberry = next(p for p in self.menu.products if p.name == "strawberry coconut")
        context = format_context([strawberry])
        self.assertIn(strawberry.issues[0], context)
        self.assertIn("21.0", context)
        self.assertIn("15.0", context)

    def test_addon_prices_and_unknown_applicability_are_grounded(self):
        text = format_addons(self.menu.addons)
        self.assertIn("extra shot: 3.0", text)
        self.assertIn("oat milk: 3.0", text)
        self.assertIn("unknown; confirm with a barista", text)

    def test_both_latte_versions_are_returned_when_search_matches_only_one(self):
        latte = next(p for p in self.menu.products if p.name == "latte")
        self.engine.index.query.return_value = SimpleNamespace(matches=[SimpleNamespace(id=latte.product_id)])
        products = retrieve("white please", self.engine.embedder, self.engine.index, self.engine.menu_by_id, 1)
        self.assertEqual({p.availability for p in products}, {"permanent", "seasonal"})
        self.assertEqual(len(products), 2)

    def test_each_turn_has_fresh_context_but_history_contains_only_dialogue(self):
        conversation = new_conversation(self.engine)
        self.assertEqual(reply(self.engine, conversation, "extras?"), "Try our latte.")
        reply(self.engine, conversation, "how much?")
        call = self.engine.genai_client.models.generate_content.call_args
        self.assertIn("extra shot: 3.0", call.kwargs["config"].system_instruction)
        self.assertEqual([c.parts[0].text for c in call.kwargs["contents"]],
                         ["extras?", "Try our latte.", "how much?"])
        self.assertTrue(all("Menu context" not in turn["text"] for turn in conversation.history))
        self.engine.genai_client.chats.create.assert_not_called()

    def test_history_is_bounded_by_turns_and_characters(self):
        conversation = new_conversation(self.engine)
        self.engine.genai_client.models.generate_content.return_value.text = "r" * 3900
        for _ in range(15):
            reply(self.engine, conversation, "q" * 4096)
        self.assertLessEqual(len(conversation.history), MAX_HISTORY_TURNS * 2)
        self.assertLessEqual(sum(len(t["text"]) for t in conversation.history), MAX_HISTORY_CHARS)
        self.assertEqual(len(conversation.history) % 2, 0)

    def test_restart_restores_bounded_state_and_followup_retrieval(self):
        conversation = new_conversation(self.engine)
        reply(self.engine, conversation, "coffee")
        restored = Conversation.restore(conversation.snapshot())
        reply(self.engine, restored, "the seasonal one")
        self.engine.embedder.encode_query.assert_called_with("Try our latte.\nthe seasonal one")

    def test_snapshot_waits_for_state_updates_and_does_not_expose_mutable_history(self):
        conversation = new_conversation(self.engine)
        started = threading.Event()
        def snapshot():
            started.set()
            return conversation.snapshot()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with conversation.lock:
                future = pool.submit(snapshot)
                self.assertTrue(started.wait(5))
                self.assertFalse(future.done())
                conversation.history = [{"role": "user", "text": "coffee"},
                                        {"role": "model", "text": "latte"}]
                conversation.context_version = "new-menu"
            state = future.result(timeout=5)
        self.assertEqual(state["context_version"], "new-menu")
        self.assertEqual(state["history"][1]["text"], "latte")
        state["history"][0]["text"] = "modified outside the lock"
        state["history"].clear()
        self.assertEqual(conversation.history[0]["text"], "coffee")
        self.assertEqual(len(conversation.history), 2)

    def test_restored_conversation_owns_its_history(self):
        state = {"history": [{"role": "user", "text": "coffee"}, {"role": "model", "text": "latte"}]}
        conversation = Conversation.restore(state)
        state["history"][0]["text"] = "mutated caller state"
        self.assertEqual(conversation.history[0]["text"], "coffee")

    def test_menu_revision_resets_stale_history(self):
        conversation = Conversation(history=[{"role": "user", "text": "old"}, {"role": "model", "text": "old price"}],
                                    last_reply="old price", context_version="old-menu")
        reply(self.engine, conversation, "price?")
        self.engine.embedder.encode_query.assert_called_once_with("price?")
        self.assertEqual(conversation.history[0]["text"], "price?")

    def test_invalid_input_or_empty_model_reply_does_not_advance_conversation(self):
        conversation = new_conversation(self.engine)
        for message in ("", " ", "x" * 4097, None):
            with self.assertRaises(ValueError):
                reply(self.engine, conversation, message)
        self.engine.genai_client.models.generate_content.assert_not_called()
        for output in (None, "", " ", "x" * 4001):
            self.engine.genai_client.models.generate_content.return_value.text = output
            with self.assertRaises(ValueError):
                reply(self.engine, conversation, "hello")
        self.assertEqual(conversation.history, [])

    def test_core_calls_are_serialized_for_shared_conversation(self):
        conversation = new_conversation(self.engine)
        entered, release = threading.Event(), threading.Event()
        calls = []
        def generate(**kwargs):
            calls.append([c.parts[0].text for c in kwargs["contents"]])
            entered.set()
            if len(calls) == 1:
                self.assertTrue(release.wait(5))
            return SimpleNamespace(text="answer")
        self.engine.genai_client.models.generate_content.side_effect = generate
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(reply, self.engine, conversation, "first")
            self.assertTrue(entered.wait(5))
            second = pool.submit(reply, self.engine, conversation, "second")
            release.set()
            first.result(timeout=5)
            second.result(timeout=5)
        self.assertEqual(calls[1], ["first", "answer", "second"])

    def test_build_engine_retains_addons_currency_and_revision(self):
        with patch("generate_prompt.chatbot.GeminiEmbedder"), patch("pinecone.Pinecone"), patch("google.genai.Client"):
            engine = build_engine(MENU_PATH, "fake", "fake", "index", None, 15, "model", MENU_PATH.with_name("absent-policies.json"))
        self.assertEqual(engine.addons, self.menu.addons)
        self.assertEqual(engine.currency, self.menu.currency)
        self.assertEqual(len(engine.context_version), 64)

    def test_truncated_model_output_is_not_committed_to_history(self):
        self.engine.genai_client.models.generate_content.return_value = SimpleNamespace(
            text="half of a reply", candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")])
        conversation = new_conversation(self.engine)
        with self.assertRaisesRegex(ValueError, "token limit"):
            reply(self.engine, conversation, "prices?")
        self.assertEqual(conversation.history, [])


if __name__ == "__main__":
    unittest.main()
