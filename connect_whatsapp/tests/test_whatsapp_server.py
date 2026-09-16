"""Offline checks for the WhatsApp webhook bridge; never touch the real
Meta Graph API, Gemini, or Pinecone."""
import hashlib
import hmac
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from connect_whatsapp.whatsapp_server import (
    REQUIRED_ENV_VARS, create_app, extract_incoming_text_message, make_app,
    send_whatsapp_message, verify_signature,
)
from generate_prompt.chatbot import ChatEngine, Conversation

APP_SECRET = "test-app-secret"
VERIFY_TOKEN = "test-verify-token"
ACCESS_TOKEN = "test-access-token"
PHONE_NUMBER_ID = "1234567890"


def sign(payload: bytes) -> str:
    return "sha256=" + hmac.new(APP_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def text_message_payload(from_number: str = "60123456789", body: str = "I want to order a coffee") -> dict:
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{"from": from_number, "type": "text", "text": {"body": body}}],
                },
            }],
        }],
    }


class VerifySignatureTests(unittest.TestCase):
    def test_matching_signature_is_accepted(self):
        payload = b'{"hello": "world"}'
        self.assertTrue(verify_signature(APP_SECRET, payload, sign(payload)))

    def test_wrong_secret_is_rejected(self):
        payload = b'{"hello": "world"}'
        bad_signature = hmac.new(b"wrong-secret", payload, hashlib.sha256).hexdigest()
        self.assertFalse(verify_signature(APP_SECRET, payload, f"sha256={bad_signature}"))

    def test_tampered_payload_is_rejected(self):
        signature = sign(b'{"hello": "world"}')
        self.assertFalse(verify_signature(APP_SECRET, b'{"hello": "mallory"}', signature))

    def test_missing_or_malformed_header_is_rejected(self):
        payload = b'{"hello": "world"}'
        self.assertFalse(verify_signature(APP_SECRET, payload, None))
        self.assertFalse(verify_signature(APP_SECRET, payload, "not-sha256-prefixed"))


class ExtractIncomingTextMessageTests(unittest.TestCase):
    def test_text_message_is_extracted(self):
        result = extract_incoming_text_message(text_message_payload("60111222333", "hello there"))
        self.assertEqual(result, ("60111222333", "hello there"))

    def test_status_callback_with_no_messages_is_ignored(self):
        payload = {"entry": [{"changes": [{"value": {"statuses": [{"status": "delivered"}]}}]}]}
        self.assertIsNone(extract_incoming_text_message(payload))

    def test_non_text_message_type_is_ignored(self):
        payload = text_message_payload()
        payload["entry"][0]["changes"][0]["value"]["messages"][0]["type"] = "image"
        self.assertIsNone(extract_incoming_text_message(payload))

    def test_malformed_payload_is_ignored_not_raised(self):
        self.assertIsNone(extract_incoming_text_message({}))
        self.assertIsNone(extract_incoming_text_message({"entry": []}))
        self.assertIsNone(extract_incoming_text_message({"entry": [{"changes": []}]}))


class SendWhatsappMessageTests(unittest.TestCase):
    def test_posts_the_expected_graph_api_request(self):
        response = Mock(status_code=200)
        with patch("connect_whatsapp.whatsapp_server.requests.post", return_value=response) as post:
            send_whatsapp_message(ACCESS_TOKEN, PHONE_NUMBER_ID, "60123456789", "Hello!")
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertIn(PHONE_NUMBER_ID, post.call_args[0][0])
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {ACCESS_TOKEN}")
        self.assertEqual(kwargs["json"], {
            "messaging_product": "whatsapp", "to": "60123456789",
            "type": "text", "text": {"body": "Hello!"},
        })

    def test_error_response_raises_instead_of_failing_silently(self):
        response = Mock(status_code=401, text="Invalid OAuth token")
        response.raise_for_status.side_effect = Exception("401")
        with patch("connect_whatsapp.whatsapp_server.requests.post", return_value=response):
            with self.assertRaises(Exception):
                send_whatsapp_message(ACCESS_TOKEN, PHONE_NUMBER_ID, "60123456789", "Hello!")


def fake_engine() -> ChatEngine:
    return ChatEngine(menu_by_id={}, embedder=Mock(), index=Mock(), top_k=15,
                       genai_client=Mock(), gemini_model="gemini-3.5-flash-lite",
                       system_instruction="You are a test bot.")


class WebhookRouteTests(unittest.TestCase):
    def setUp(self):
        self.engine = fake_engine()
        self.app = create_app(self.engine, ACCESS_TOKEN, PHONE_NUMBER_ID, VERIFY_TOKEN, APP_SECRET)
        self.client = self.app.test_client()

    def test_get_webhook_with_correct_token_echoes_challenge(self):
        response = self.client.get("/webhook", query_string={
            "hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "12345",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "12345")

    def test_get_webhook_with_wrong_token_is_forbidden(self):
        response = self.client.get("/webhook", query_string={
            "hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "12345",
        })
        self.assertEqual(response.status_code, 403)

    def test_post_webhook_without_valid_signature_is_forbidden(self):
        body = json.dumps(text_message_payload()).encode("utf-8")
        response = self.client.post("/webhook", data=body, content_type="application/json",
                                     headers={"X-Hub-Signature-256": "sha256=wrong"})
        self.assertEqual(response.status_code, 403)

    def test_post_webhook_with_no_new_message_is_acknowledged_and_ignored(self):
        payload = {"entry": [{"changes": [{"value": {"statuses": [{"status": "read"}]}}]}]}
        body = json.dumps(payload).encode("utf-8")
        response = self.client.post("/webhook", data=body, content_type="application/json",
                                     headers={"X-Hub-Signature-256": sign(body)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ignored")

    def test_valid_text_message_gets_a_reply_sent_back(self):
        body = json.dumps(text_message_payload("60123456789", "I want to order a coffee")).encode("utf-8")
        with patch("connect_whatsapp.whatsapp_server.new_conversation") as mock_new_conv, \
             patch("connect_whatsapp.whatsapp_server.reply", return_value="Sure, what would you like?") as mock_reply, \
             patch("connect_whatsapp.whatsapp_server.send_whatsapp_message") as mock_send:
            mock_new_conv.return_value = Conversation(chat=Mock())
            response = self.client.post("/webhook", data=body, content_type="application/json",
                                         headers={"X-Hub-Signature-256": sign(body)})
        self.assertEqual(response.status_code, 200)
        mock_reply.assert_called_once()
        self.assertEqual(mock_reply.call_args[0][2], "I want to order a coffee")
        mock_send.assert_called_once_with(ACCESS_TOKEN, PHONE_NUMBER_ID, "60123456789",
                                           "Sure, what would you like?")

    def test_same_phone_number_reuses_its_conversation_across_messages(self):
        with patch("connect_whatsapp.whatsapp_server.new_conversation") as mock_new_conv, \
             patch("connect_whatsapp.whatsapp_server.reply", return_value="ok") as mock_reply, \
             patch("connect_whatsapp.whatsapp_server.send_whatsapp_message"):
            mock_new_conv.return_value = Conversation(chat=Mock())
            for _ in range(2):
                body = json.dumps(text_message_payload("60123456789", "hi")).encode("utf-8")
                self.client.post("/webhook", data=body, content_type="application/json",
                                  headers={"X-Hub-Signature-256": sign(body)})
        # A brand-new Conversation is only created once; the second message
        # for the same customer reuses it (same object passed to reply()).
        mock_new_conv.assert_called_once()
        first_conversation = mock_reply.call_args_list[0][0][1]
        second_conversation = mock_reply.call_args_list[1][0][1]
        self.assertIs(first_conversation, second_conversation)

    def test_reply_failure_still_returns_200_with_a_fallback_message(self):
        body = json.dumps(text_message_payload("60123456789", "hi")).encode("utf-8")
        with patch("connect_whatsapp.whatsapp_server.new_conversation") as mock_new_conv, \
             patch("connect_whatsapp.whatsapp_server.reply", side_effect=RuntimeError("boom")), \
             patch("connect_whatsapp.whatsapp_server.send_whatsapp_message") as mock_send:
            mock_new_conv.return_value = Conversation(chat=Mock())
            response = self.client.post("/webhook", data=body, content_type="application/json",
                                         headers={"X-Hub-Signature-256": sign(body)})
        self.assertEqual(response.status_code, 200)
        mock_send.assert_called_once()
        self.assertIn("barista", mock_send.call_args[0][3].lower())

    def test_health_endpoint_reports_active_conversation_count(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["active_conversations"], 0)


def all_required_env(**overrides) -> dict:
    env = {name: f"fake-{name.lower()}" for name in REQUIRED_ENV_VARS}
    env.update(overrides)
    return env


class MakeAppTests(unittest.TestCase):
    """make_app() is the zero-argument factory a production WSGI server
    (gunicorn on Render) imports and calls -- these confirm it reads
    config from the environment the way connect_whatsapp/whatsapp-
    integration.md documents, without making any real network call."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.menu_path = Path(directory.name, "structure.json")
        self.menu_path.write_text("{}", encoding="utf-8")

    def test_missing_env_var_raises_a_clear_error(self):
        env = all_required_env()
        del env["WHATSAPP_APP_SECRET"]
        env["MENU_FILE"] = str(self.menu_path)
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "WHATSAPP_APP_SECRET"):
                make_app()

    def test_missing_menu_file_raises_a_clear_error(self):
        env = all_required_env(MENU_FILE=str(self.menu_path.with_name("nope.json")))
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Menu file not found"):
                make_app()

    def test_valid_config_builds_the_app_from_env_vars(self):
        env = all_required_env(MENU_FILE=str(self.menu_path), PINECONE_INDEX="custom-index",
                                CHATBOT_TOP_K="7")
        with patch.dict("os.environ", env, clear=True), \
             patch("connect_whatsapp.whatsapp_server.build_engine") as mock_build_engine:
            mock_build_engine.return_value = ChatEngine(
                menu_by_id={}, embedder=Mock(), index=Mock(), top_k=7,
                genai_client=Mock(), gemini_model="gemini-3.5-flash-lite", system_instruction="x",
            )
            app = make_app()
        self.assertIsNotNone(app)
        mock_build_engine.assert_called_once()
        self.assertEqual(mock_build_engine.call_args.kwargs["index_name"], "custom-index")
        self.assertEqual(mock_build_engine.call_args.kwargs["top_k"], 7)


if __name__ == "__main__":
    unittest.main()
