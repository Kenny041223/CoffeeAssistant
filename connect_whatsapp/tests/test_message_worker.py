"""Real SQLite transactions with fake model/delivery boundaries."""
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

from connect_whatsapp.message_store import (
    DEDUP_RETENTION, IncomingMessage, LEASE_SECONDS, LeaseLost,
    MAX_ATTEMPTS, MessageStore, SESSION_TTL,
)
from connect_whatsapp.message_worker import MessageWorker
from connect_whatsapp.whatsapp_server import DeliveryError


class WorkerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "queue.sqlite3"
        self.now = 1000000.0
        self.store = MessageStore(self.path, clock=lambda: self.now)
        self.send = Mock(return_value="provider-id")
        self.generate = Mock(side_effect=self.answer)
        self.worker = self.new_worker()

    @staticmethod
    def answer(engine, conversation, message):
        text = "answer to " + message
        conversation.history += [{"role": "user", "text": message}, {"role": "model", "text": text}]
        conversation.last_reply = text
        return text

    def new_worker(self, store=None):
        return MessageWorker(store or self.store, Mock(), "token", "12345", generate=self.generate, send=self.send)

    def enqueue(self, mid="one", sender="60123", text="coffee", timestamp=None):
        self.store.enqueue([IncomingMessage(mid, sender, text, self.now if timestamp is None else timestamp)])

    def row(self, mid="one"):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            return dict(db.execute("SELECT * FROM messages WHERE message_id=?", (mid,)).fetchone())

    def test_duplicate_after_success_does_not_send_or_generate_again(self):
        self.enqueue()
        self.assertTrue(self.worker.process_one())
        self.enqueue()
        self.assertFalse(self.worker.process_one())
        self.generate.assert_called_once()
        self.send.assert_called_once()
        self.assertEqual(self.row()["text"], "")

    def test_failed_delivery_survives_restart_and_reuses_saved_reply(self):
        self.enqueue()
        self.send.side_effect = DeliveryError("Graph HTTP 503", retryable=True)
        self.worker.process_one()
        self.assertEqual(self.row()["status"], "pending")
        self.assertEqual(self.store.session("60123"), {})
        self.assertFalse(self.worker.process_one())  # backoff
        self.now += 3
        reopened = MessageStore(self.path, clock=lambda: self.now)
        self.send.side_effect = None
        self.new_worker(reopened).process_one()
        self.generate.assert_called_once()
        self.assertEqual(self.send.call_count, 2)
        self.assertEqual(self.row()["status"], "sent")
        self.assertEqual(len(reopened.session("60123")["history"]), 2)

    def test_crashed_worker_is_recovered_and_stale_token_is_fenced(self):
        self.enqueue()
        abandoned = self.store.claim()
        self.store.save_reply(abandoned, "saved reply", {"history": [], "context_version": ""})
        self.assertIsNone(self.store.claim())
        self.now += LEASE_SECONDS + 1
        replacement = self.store.claim()
        self.assertNotEqual(abandoned.token, replacement.token)
        for operation in (lambda: self.store.renew(abandoned), lambda: self.store.complete(abandoned),
                          lambda: self.store.save_reply(abandoned, "stale", {}),
                          lambda: self.store.fail(abandoned, "stale")):
            with self.assertRaises(LeaseLost):
                operation()
        self.assertEqual(replacement.reply_text, "saved reply")
        self.store.complete(replacement)

    def test_lease_renewal_prevents_another_worker_claim(self):
        self.enqueue()
        job = self.store.claim()
        self.now += LEASE_SECONDS - 1
        self.store.renew(job)
        self.now += 2
        self.assertIsNone(self.store.claim())

    def test_same_customer_waits_during_backoff_other_customers_continue(self):
        self.enqueue("one", text="first")
        self.enqueue("two", text="second")
        self.enqueue("three", sender="60999", text="other")
        first = self.store.claim()
        self.store.fail(first, "temporary")
        self.assertEqual(self.store.claim().message_id, "three")
        self.assertIsNone(self.store.claim())
        self.now += 3
        self.assertEqual(self.store.claim().message_id, "one")

    def test_two_workers_never_generate_for_same_customer_concurrently(self):
        self.enqueue("one", text="first")
        self.enqueue("two", text="second")
        entered, release = threading.Event(), threading.Event()
        def generate(engine, conversation, text):
            entered.set()
            self.assertTrue(release.wait(5))
            return self.answer(engine, conversation, text)
        self.generate.side_effect = generate
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.worker.process_one)
            self.assertTrue(entered.wait(5))
            second_worker = self.new_worker(MessageStore(self.path, clock=lambda: self.now))
            try:
                self.assertFalse(second_worker.process_one())
            finally:
                release.set()
            self.assertTrue(first.result(timeout=5))
        self.generate.side_effect = self.answer
        self.worker.process_one()
        second_conversation = self.generate.call_args[0][1]
        self.assertEqual(second_conversation.history[0]["text"], "first")
        self.assertEqual(second_conversation.history[2]["text"], "second")

    def test_separate_processes_cannot_claim_same_customer(self):
        self.enqueue("one")
        self.enqueue("two")
        script = ("from pathlib import Path; import sys; "
                  "from connect_whatsapp.message_store import MessageStore; "
                  "s=MessageStore(Path(sys.argv[1]),clock=lambda:1000000.0); "
                  "j=s.claim(); print(j.message_id if j else 'none')")
        processes = [subprocess.Popen([sys.executable, "-c", script, str(self.path)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        outputs = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            outputs.append(stdout.strip())
        self.assertCountEqual(outputs, ["one", "none"])

    def test_generation_failure_retries_without_advancing_history(self):
        self.enqueue()
        self.generate.side_effect = RuntimeError("provider response with secret")
        self.worker.process_one()
        self.send.assert_not_called()
        self.assertEqual(self.store.session("60123"), {})
        self.assertEqual(self.row()["error"], "RuntimeError")
        self.now += 3
        self.generate.side_effect = self.answer
        self.worker.process_one()
        self.assertEqual(self.row()["status"], "sent")

    def test_permanent_failure_is_visible_and_safe_manual_retry_is_supported(self):
        self.enqueue()
        self.send.side_effect = DeliveryError("Graph HTTP 401", retryable=False)
        self.worker.process_one()
        self.assertEqual(self.store.stats()["failed"], 1)
        self.store.retry_failed("one")
        self.send.side_effect = None
        self.worker.process_one()
        self.generate.assert_called_once()
        self.assertEqual(self.row()["status"], "sent")

    def test_manual_retry_cannot_rewind_newer_conversations(self):
        self.enqueue()
        self.store.fail(self.store.claim(), "failure", retryable=False)
        self.enqueue("two")
        with self.assertRaisesRegex(ValueError, "newer"):
            self.store.retry_failed("one")

    def test_retry_budget_is_bounded(self):
        self.enqueue()
        self.send.side_effect = DeliveryError("Graph HTTP 429", retryable=True)
        for _ in range(MAX_ATTEMPTS):
            self.worker.process_one()
            self.now += 301
        self.assertEqual(self.row()["status"], "failed")
        self.assertFalse(self.worker.process_one())
        self.assertEqual(self.send.call_count, MAX_ATTEMPTS)
        self.generate.assert_called_once()

    def test_old_messages_are_retained_as_failed_without_sending(self):
        self.enqueue(timestamp=self.now - SESSION_TTL - 1)
        self.worker.process_one()
        self.send.assert_not_called()
        self.generate.assert_not_called()
        self.assertEqual(self.store.stats()["failed"], 1)
        with self.assertRaisesRegex(ValueError, "expired"):
            self.store.retry_failed("one")

    def test_session_expiry_and_retention_cleanup(self):
        self.enqueue()
        self.worker.process_one()
        self.assertTrue(self.store.session("60123"))
        self.now += SESSION_TTL + 1
        self.store.cleanup()
        self.assertEqual(self.store.session("60123"), {})
        self.assertIsNotNone(self.row())  # deduplication outlives dialogue
        self.now += DEDUP_RETENTION
        self.store.cleanup()
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)

    def test_queue_batch_rolls_back_on_storage_constraint_failure(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.enqueue([IncomingMessage("one", "60123", "valid", self.now),
                                IncomingMessage("two", "60123", None, self.now)])
        self.assertEqual(self.store.stats()["pending"], 0)

    def test_database_cannot_be_reused_by_another_business_number(self):
        with self.assertRaisesRegex(ValueError, "different WhatsApp"):
            MessageWorker(self.store, Mock(), "token", "another-number")

    def test_generation_cannot_send_after_service_window_expires(self):
        self.enqueue(timestamp=self.now - SESSION_TTL + 1)
        def slow_generation(engine, conversation, text):
            self.now += 2
            return self.answer(engine, conversation, text)
        self.generate.side_effect = slow_generation
        self.worker.process_one()
        self.send.assert_not_called()
        self.assertEqual(self.store.stats()["failed"], 1)

    def test_signed_batch_through_real_chatbot_with_delivery_retry(self):
        import json
        from types import SimpleNamespace
        from generate_prompt.chatbot import ChatEngine, reply
        from connect_whatsapp.whatsapp_server import create_app
        from connect_whatsapp.tests.test_whatsapp_server import payload, sign, SECRET, PHONE_ID
        # Match the webhook parser's real timestamp while retaining a controllable worker clock.
        import time
        self.now = time.time()
        engine = ChatEngine(menu_by_id={}, embedder=Mock(), index=Mock(), top_k=15,
                            genai_client=Mock(), gemini_model="test", system_instruction="test")
        engine.index.query.return_value = SimpleNamespace(matches=[])
        engine.genai_client.models.generate_content.return_value = SimpleNamespace(text="Which coffee?")
        app = create_app(self.store, PHONE_ID, "verify", SECRET)
        raw = json.dumps(payload("one", "two")).encode()
        with app.test_client() as client:
            for _ in range(2):
                result = client.post("/webhook", data=raw, content_type="application/json",
                                     headers={"X-Hub-Signature-256": sign(raw)})
                self.assertEqual(result.status_code, 200)
        self.send.side_effect = [DeliveryError("Graph HTTP 503", retryable=True), "id-one", "id-two"]
        worker = MessageWorker(self.store, engine, "token", PHONE_ID, generate=reply, send=self.send)
        worker.process_one()
        self.now += 3
        worker.process_one()
        worker.process_one()
        self.assertFalse(worker.process_one())
        self.assertEqual(engine.genai_client.models.generate_content.call_count, 2)
        self.assertEqual(self.send.call_count, 3)
        self.assertEqual(self.row("one")["status"], "sent")
        self.assertEqual(self.row("two")["status"], "sent")
        self.assertEqual(len(self.store.session("60123456789")["history"]), 4)


if __name__ == "__main__":
    unittest.main()
