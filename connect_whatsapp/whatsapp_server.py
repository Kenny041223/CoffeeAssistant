"""Signed WhatsApp webhooks: validate, persist, acknowledge.

Run the separate message_worker process against the same local SQLite database
to generate and send replies. Web workers never perform model or Graph API calls.
"""
import argparse
import hashlib
import hmac
import logging
import math
import os
import re
import sqlite3
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, request

from connect_whatsapp.message_store import IncomingMessage, MessageStore
from generate_prompt.chatbot import MAX_MESSAGE_CHARS, MAX_REPLY_CHARS

GRAPH_API_VERSION = "v21.0"
REQUEST_TIMEOUT_SECONDS = 30
REQUIRED_ENV_VARS = ("WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET")
logger = logging.getLogger(__name__)


def require_env(names: tuple[str, ...]) -> dict[str, str]:
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")
    return values


def store_from_env() -> MessageStore:
    return MessageStore(Path(os.environ.get("WHATSAPP_DB_PATH", "connect_whatsapp/data/messages.sqlite3")))


def verify_signature(app_secret: str, payload: bytes, signature_header: str | None) -> bool:
    if not signature_header or not re.fullmatch(r"sha256=[0-9a-fA-F]{64}", signature_header):
        return False
    expected = hmac.new(app_secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header[7:].lower())


def extract_incoming_text_messages(payload: dict, phone_number_id: str) -> list[IncomingMessage]:
    """Visit every entry/change/message; reject malformed text batches atomically."""
    if not isinstance(payload, dict) or payload.get("object") != "whatsapp_business_account":
        raise ValueError("Expected a WhatsApp Business webhook object")
    entries = payload.get("entry")
    if not isinstance(entries, list):
        raise ValueError("Expected an entry array")
    result = []
    for entry in entries:
        changes = entry.get("changes") if isinstance(entry, dict) else None
        if not isinstance(changes, list):
            raise ValueError("Expected a changes array")
        for change in changes:
            if not isinstance(change, dict):
                raise ValueError("Invalid webhook change")
            if change.get("field") != "messages":
                continue
            value = change.get("value")
            if not isinstance(value, dict):
                raise ValueError("Invalid messages value")
            metadata = value.get("metadata", {})
            if not isinstance(metadata, dict) or metadata.get("phone_number_id") != phone_number_id:
                continue
            messages = value.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError("Expected a messages array")
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("Invalid message")
                if message.get("type") != "text":
                    continue
                mid, sender = message.get("id"), message.get("from")
                text_object = message.get("text")
                body = text_object.get("body") if isinstance(text_object, dict) else None
                if (not isinstance(mid, str) or not mid.strip() or len(mid) > 512
                        or not isinstance(sender, str) or not re.fullmatch(r"[0-9]{1,32}", sender)
                        or not isinstance(body, str) or not body.strip() or len(body) > MAX_MESSAGE_CHARS):
                    raise ValueError("Invalid text message fields")
                try:
                    timestamp = float(message["timestamp"])
                except (KeyError, ValueError, TypeError):
                    raise ValueError("Invalid message timestamp") from None
                if not math.isfinite(timestamp) or timestamp < 0 or timestamp > time.time() + 300:
                    raise ValueError("Invalid message timestamp")
                result.append(IncomingMessage(mid, sender, body, timestamp))
    return sorted(result, key=lambda message: message.timestamp)


class DeliveryError(RuntimeError):
    def __init__(self, reason: str, *, retryable: bool):
        super().__init__(reason)
        self.retryable = retryable


def send_whatsapp_message(access_token: str, phone_number_id: str, to: str, text: str) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_REPLY_CHARS:
        raise DeliveryError("Invalid outgoing text", retryable=False)
    version = os.environ.get("WHATSAPP_GRAPH_VERSION", GRAPH_API_VERSION)
    if not re.fullmatch(r"v[0-9]+\.[0-9]+", version):
        raise DeliveryError("Invalid Graph API version", retryable=False)
    try:
        response = requests.post(
            f"https://graph.facebook.com/{version}/{phone_number_id}/messages",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}},
            timeout=(5, REQUEST_TIMEOUT_SECONDS),
        )
    except requests.RequestException:
        raise DeliveryError("Graph transport failure", retryable=True) from None
    if response.status_code >= 400:
        raise DeliveryError(f"Graph HTTP {response.status_code}",
                            retryable=response.status_code in (408, 429) or response.status_code >= 500)
    try:
        message_id = response.json()["messages"][0]["id"]
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("Missing message ID")
    except (ValueError, KeyError, IndexError, TypeError):
        raise DeliveryError("Graph returned no message ID", retryable=True) from None
    return message_id


def create_app(store: MessageStore, phone_number_id: str, verify_token: str, app_secret: str) -> Flask:
    if not all((phone_number_id, verify_token, app_secret)):
        raise ValueError("WhatsApp webhook configuration cannot be empty")
    store.bind_phone_number(phone_number_id)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024

    @app.get("/webhook")
    def verify_webhook():
        token = request.args.get("hub.verify_token", "")
        if (request.args.get("hub.mode") == "subscribe"
                and hmac.compare_digest(token.encode(), verify_token.encode())):
            return request.args.get("hub.challenge", ""), 200
        return "Forbidden", 403

    @app.post("/webhook")
    def incoming_message():
        if not verify_signature(app_secret, request.get_data(), request.headers.get("X-Hub-Signature-256")):
            return "Forbidden", 403
        try:
            messages = extract_incoming_text_messages(request.get_json(silent=True), phone_number_id)
        except ValueError:
            return jsonify(error="Invalid webhook payload"), 400
        try:
            accepted = store.enqueue(messages)
        except sqlite3.Error:
            logger.error("Webhook persistence failed")
            return jsonify(error="Queue unavailable"), 503
        return jsonify(status="queued" if messages else "ignored", accepted=accepted), 200

    @app.get("/health")
    def health():
        return jsonify(status="ok"), 200

    @app.get("/ready")
    def ready():
        try:
            stats = store.stats()
        except sqlite3.Error:
            return jsonify(status="unavailable"), 503
        ready_to_accept = stats["workers"] > 0
        healthy = ready_to_accept and stats["failed"] == 0
        return jsonify(status="ok" if healthy else "degraded", **stats), 200 if ready_to_accept else 503

    return app


def make_app() -> Flask:
    """gunicorn 'connect_whatsapp.whatsapp_server:make_app()' --bind 0.0.0.0:$PORT"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    env = require_env(REQUIRED_ENV_VARS)
    return create_app(store_from_env(), env["WHATSAPP_PHONE_NUMBER_ID"],
                      env["WHATSAPP_VERIFY_TOKEN"], env["WHATSAPP_APP_SECRET"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    make_app().run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
