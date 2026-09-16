"""Webhook and Graph API boundary tests; no external services."""
import hashlib
import hmac
import json
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from connect_whatsapp.message_store import MessageStore
from connect_whatsapp.whatsapp_server import (
    DeliveryError, REQUIRED_ENV_VARS, create_app,
    make_app, send_whatsapp_message, verify_signature,
)

SECRET = "test-secret"
PHONE_ID = "12345"


def payload(*ids):
    return {"object": "whatsapp_business_account", "entry": [{"changes": [{"field": "messages", "value": {
        "metadata": {"phone_number_id": PHONE_ID},
        "messages": [{"id": mid, "from": "60123456789", "timestamp": str(int(time.time())),
                      "type": "text", "text": {"body": "coffee please"}} for mid in ids],
    }}]}]}


def sign(body):
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


class WebhookTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "messages.sqlite3"
        self.store = MessageStore(self.path)
        self.app = create_app(self.store, PHONE_ID, "verify", SECRET)

    def post(self, value):
        body = json.dumps(value).encode()
        with self.app.test_client() as client:
            return client.post("/webhook", data=body, content_type="application/json",
                               headers={"X-Hub-Signature-256": sign(body)})

    def test_signature_rejects_tampering_missing_and_unicode_headers(self):
        self.assertTrue(verify_signature(SECRET, b"ok", sign(b"ok")))
        for header in (None, "bad", "sha256=" + "x" * 64, "sha256=" + "\u00e9" * 64, sign(b"other")):
            self.assertFalse(verify_signature(SECRET, b"ok", header))

    def test_handshake_requires_token(self):
        with self.app.test_client() as client:
            args = {"hub.mode": "subscribe", "hub.verify_token": "verify", "hub.challenge": "challenge"}
            self.assertEqual(client.get("/webhook", query_string=args).data, b"challenge")
            args["hub.verify_token"] = "wrong"
            self.assertEqual(client.get("/webhook", query_string=args).status_code, 403)

    def test_unsigned_message_is_not_enqueued(self):
        with self.app.test_client() as client:
            self.assertEqual(client.post("/webhook", json=payload("one")).status_code, 403)
        self.assertEqual(self.store.stats()["pending"], 0)

    def test_acknowledges_only_after_durable_enqueue_without_network_calls(self):
        with patch("requests.post", side_effect=AssertionError("No network in webhook")):
            result = self.post(payload("one"))
        self.assertEqual(result.status_code, 200)
        reopened = MessageStore(self.path)
        self.assertEqual(reopened.claim().message_id, "one")

    def test_all_entries_changes_and_messages_are_enqueued(self):
        data = payload("one", "two")
        data["entry"][0]["changes"].extend(payload("three")["entry"][0]["changes"])
        data["entry"].extend(payload("four")["entry"])
        result = self.post(data)
        self.assertEqual(result.json["accepted"], 4)
        self.assertEqual(self.store.stats()["pending"], 4)

    def test_concurrent_redeliveries_are_deduplicated(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.post(payload("same")), range(8)))
        self.assertTrue(all(r.status_code == 200 for r in results))
        self.assertEqual(sum(r.json["accepted"] for r in results), 1)

    def test_non_text_receipts_and_other_phone_numbers_are_ignored(self):
        for variant in ("image", "receipt", "other-phone"):
            with self.subTest(variant=variant):
                data = payload("one")
                value = data["entry"][0]["changes"][0]["value"]
                if variant == "image":
                    value["messages"][0]["type"] = "image"
                elif variant == "receipt":
                    del value["messages"]
                    value["statuses"] = [{"status": "read"}]
                else:
                    value["metadata"]["phone_number_id"] = "another"
                self.assertEqual(self.post(data).json["status"], "ignored")
        self.assertEqual(self.store.stats()["pending"], 0)

    def test_malformed_text_batch_is_rejected_without_partial_acceptance(self):
        data = payload("good", "bad")
        del data["entry"][0]["changes"][0]["value"]["messages"][1]["id"]
        self.assertEqual(self.post(data).status_code, 400)
        self.assertEqual(self.store.stats()["pending"], 0)
        for invalid in (None, [], {}, {"entry": [None]}, {"object": "whatsapp_business_account", "entry": [None]}):
            self.assertEqual(self.post(invalid).status_code, 400)

    def test_storage_failure_is_not_acknowledged_as_success(self):
        with patch.object(self.store, "enqueue", side_effect=sqlite3.OperationalError("disk full")):
            self.assertEqual(self.post(payload("one")).status_code, 503)

    def test_request_and_message_limits(self):
        data = payload("one")
        data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"] = "x" * 4097
        self.assertEqual(self.post(data).status_code, 400)
        with self.app.test_client() as client:
            self.assertEqual(client.post("/webhook", data=b"x" * (256 * 1024 + 1)).status_code, 413)

    def test_readiness_detects_missing_worker_and_failed_jobs(self):
        with self.app.test_client() as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 503)
            self.store.heartbeat("worker")
            self.assertEqual(client.get("/ready").status_code, 200)
            self.post(payload("one"))
            self.store.fail(self.store.claim(), "permanent failure", retryable=False)
            self.assertEqual(client.get("/ready").json["failed"], 1)
            self.assertEqual(client.get("/ready").json["status"], "degraded")
            self.assertEqual(client.get("/ready").status_code, 200)

    def test_factory_needs_only_webhook_credentials_and_persistent_path(self):
        env = {name: "test-value" for name in REQUIRED_ENV_VARS}
        env["WHATSAPP_PHONE_NUMBER_ID"] = PHONE_ID
        env["WHATSAPP_DB_PATH"] = str(self.path)
        with patch.dict("os.environ", env, clear=True):
            self.assertIsNotNone(make_app())
        del env["WHATSAPP_APP_SECRET"]
        with patch.dict("os.environ", env, clear=True), self.assertRaisesRegex(RuntimeError, "APP_SECRET"):
            make_app()


class SendTests(unittest.TestCase):
    def test_success_validates_provider_id_and_sets_timeout(self):
        response = Mock(status_code=200)
        response.json.return_value = {"messages": [{"id": "outbound"}]}
        with patch("connect_whatsapp.whatsapp_server.requests.post", return_value=response) as post:
            self.assertEqual(send_whatsapp_message("token", PHONE_ID, "60123", "hello"), "outbound")
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 30))
        self.assertEqual(post.call_args.kwargs["json"]["text"]["body"], "hello")

    def test_transient_and_permanent_failures_are_distinguished(self):
        for status in (400, 401, 403, 408, 429, 500, 503):
            with self.subTest(status=status), patch("connect_whatsapp.whatsapp_server.requests.post",
                                                  return_value=Mock(status_code=status)):
                with self.assertRaises(DeliveryError) as exc:
                    send_whatsapp_message("token", PHONE_ID, "60123", "hello")
                self.assertEqual(exc.exception.retryable, status in (408, 429) or status >= 500)

    def test_timeouts_and_invalid_success_payloads_are_retryable(self):
        with patch("connect_whatsapp.whatsapp_server.requests.post", side_effect=requests.Timeout("secret")):
            with self.assertRaises(DeliveryError) as exc:
                send_whatsapp_message("token", PHONE_ID, "60123", "hello")
        self.assertTrue(exc.exception.retryable)
        self.assertNotIn("secret", str(exc.exception))
        response = Mock(status_code=200)
        response.json.return_value = {}
        with patch("connect_whatsapp.whatsapp_server.requests.post", return_value=response):
            with self.assertRaises(DeliveryError):
                send_whatsapp_message("token", PHONE_ID, "60123", "hello")


if __name__ == "__main__":
    unittest.main()
