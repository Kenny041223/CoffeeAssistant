"""Bridges WhatsApp Cloud API (Meta's official Business Platform) to the
existing chatbot core in generate_prompt/chatbot.py -- no reply logic is
duplicated here, this module only handles the WhatsApp-specific plumbing:

- Meta calls this server's POST /webhook for every inbound customer message.
- This server calls Meta's Graph API to send the reply back.
- One Conversation (see chatbot.py) is kept in memory per customer phone
  number, so each customer's chat history and retrieval context stay
  independent of every other customer's.

Requires a WhatsApp Business app already set up in Meta's developer console
(App ID, phone number ID, permanent access token) and a public HTTPS URL
Meta can reach -- see connect_whatsapp/whatsapp-integration.md for the full
walkthrough. This module never starts a server on import; call main() or
create_app().

In-memory state only: conversations dict is lost on restart, and this
won't share state across multiple worker processes. Fine for a single-
process demo/portfolio deployment; see connect_whatsapp/whatsapp-integration.md
"Still open" for what a production deployment would still need.
"""
import argparse
import hashlib
import hmac
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, request

from generate_prompt.chatbot import ChatEngine, Conversation, build_engine, new_conversation, reply

GRAPH_API_VERSION = "v21.0"
REQUEST_TIMEOUT_SECONDS = 30

logger = logging.getLogger("whatsapp_server")


def verify_signature(app_secret: str, payload: bytes, signature_header: str | None) -> bool:
    """Meta signs every webhook POST with X-Hub-Signature-256 (HMAC-SHA256
    over the exact raw request body, keyed by the app secret). Reject
    anything that doesn't match rather than trust the network -- otherwise
    anyone who finds this URL could feed it fake "messages" and burn the
    Gemini/Pinecone quota, or worse, inject content that gets echoed back
    to a real customer."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    provided = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, provided)


def extract_incoming_text_message(payload: dict) -> tuple[str, str] | None:
    """Pull (customer_phone_number, message_text) out of one WhatsApp Cloud
    API webhook payload. Returns None for anything that isn't a new text
    message -- delivery/read status callbacks, images/audio/etc, or a
    malformed payload -- since there's nothing to reply to in those cases."""
    try:
        value = payload["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")
        if not messages:
            return None  # e.g. a status callback (sent/delivered/read), not a new message
        message = messages[0]
        if message.get("type") != "text":
            return None  # images/audio/location/etc -- out of scope for now
        return message["from"], message["text"]["body"]
    except (KeyError, IndexError, TypeError):
        return None


def send_whatsapp_message(access_token: str, phone_number_id: str, to: str, text: str) -> None:
    """Send one text reply via the Cloud API. Raises on failure -- the
    caller decides whether/how to retry; this function never swallows an
    error silently, so a broken send doesn't look like a broken bot."""
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/messages"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        logger.error("WhatsApp send to %s failed (%s): %s", to, response.status_code, response.text)
        response.raise_for_status()


def create_app(engine: ChatEngine, access_token: str, phone_number_id: str,
               verify_token: str, app_secret: str) -> Flask:
    """Builds the Flask app. Kept separate from main() so tests can build
    one against a fake ChatEngine without touching argv/env/real network
    clients."""
    app = Flask(__name__)
    conversations: dict[str, Conversation] = {}

    @app.get("/webhook")
    def verify_webhook():
        # Meta's one-time setup handshake: it calls this with a token you
        # chose when configuring the webhook, and expects the challenge
        # echoed back verbatim if (and only if) the token matches.
        if (request.args.get("hub.mode") == "subscribe"
                and request.args.get("hub.verify_token") == verify_token):
            return request.args.get("hub.challenge", ""), 200
        return "Forbidden", 403

    @app.post("/webhook")
    def incoming_message():
        if not verify_signature(app_secret, request.get_data(), request.headers.get("X-Hub-Signature-256")):
            logger.warning("Rejected webhook POST with an invalid or missing signature")
            return "Forbidden", 403

        payload = request.get_json(silent=True) or {}
        extracted = extract_incoming_text_message(payload)
        if extracted is None:
            # Not an error -- Meta also posts delivery/read receipts and
            # other event types to this same endpoint. Acknowledge with 200
            # so Meta doesn't retry-storm us; there's just nothing to reply to.
            return jsonify(status="ignored"), 200

        from_number, text = extracted
        if from_number not in conversations:
            conversations[from_number] = new_conversation(engine)

        try:
            reply_text = reply(engine, conversations[from_number], text)
        except Exception:
            logger.exception("Failed to generate a reply for %s", from_number)
            reply_text = ("Sorry, something went wrong on our end -- please try again in "
                          "a moment, or ask a barista in the shop.")

        try:
            send_whatsapp_message(access_token, phone_number_id, from_number, reply_text)
        except Exception:
            logger.exception("Failed to send the WhatsApp reply to %s", from_number)
            # The customer's message was already answered by the model; a
            # delivery failure here is a network/Meta-side problem, not
            # something retrying this webhook call would fix. Meta expects
            # 200 regardless, or it will retry the whole incoming event.

        return jsonify(status="ok"), 200

    @app.get("/health")
    def health():
        return jsonify(status="ok", active_conversations=len(conversations)), 200

    return app


REQUIRED_ENV_VARS = (
    "PINECONE_API_KEY", "GEMINI_API_KEY", "WHATSAPP_ACCESS_TOKEN",
    "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET",
)


def _build_flask_app(menu_path: Path, index_name: str, namespace: str | None, top_k: int,
                      gemini_model: str, shop_policies_path: Path) -> Flask:
    """Shared by both entry points below: validate config, build the engine,
    wrap it in a Flask app. Raises on missing/invalid config rather than
    returning an error code, since make_app() (used by a WSGI server that
    imported this module) has no return-code channel to report through --
    a raised exception there fails the server's startup loudly, which is
    what should happen for a misconfigured deployment."""
    if not menu_path.is_file():
        raise RuntimeError(f"Menu file not found: {menu_path}. Run menu generation first.")
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "See connect_whatsapp/whatsapp-integration.md for where each one comes from."
        )

    engine = build_engine(
        menu_path=menu_path,
        pinecone_key=os.environ["PINECONE_API_KEY"],
        gemini_key=os.environ["GEMINI_API_KEY"],
        index_name=index_name,
        namespace=namespace,
        top_k=top_k,
        gemini_model=gemini_model,
        shop_policies_path=shop_policies_path,
    )
    logger.info("Ready (%d products indexed).", len(engine.menu_by_id))
    return create_app(
        engine=engine,
        access_token=os.environ["WHATSAPP_ACCESS_TOKEN"],
        phone_number_id=os.environ["WHATSAPP_PHONE_NUMBER_ID"],
        verify_token=os.environ["WHATSAPP_VERIFY_TOKEN"],
        app_secret=os.environ["WHATSAPP_APP_SECRET"],
    )


def make_app() -> Flask:
    """Zero-argument app factory for a real WSGI server in production, e.g.
    on Render:
        gunicorn 'connect_whatsapp.whatsapp_server:make_app()' --bind 0.0.0.0:$PORT
    The trailing () in the app string is gunicorn's own syntax for "call
    this as a factory" -- gunicorn has no --factory flag (that's a
    different tool's convention, e.g. uvicorn's). Quote the whole app
    string in a shell so the parentheses aren't interpreted by the shell.
    A WSGI server imports this module and calls this function itself --
    there's no argv to parse, so every setting comes from an environment
    variable instead of a CLI flag (same defaults as main()'s flags below).
    Configure the six REQUIRED_ENV_VARS above, plus optionally MENU_FILE,
    PINECONE_INDEX, PINECONE_NAMESPACE, CHATBOT_TOP_K, GEMINI_MODEL, and
    SHOP_POLICIES_FILE to override their defaults."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return _build_flask_app(
        menu_path=Path(os.environ.get("MENU_FILE", "structure.json")),
        index_name=os.environ.get("PINECONE_INDEX", "coffee-menu"),
        namespace=os.environ.get("PINECONE_NAMESPACE") or None,
        top_k=int(os.environ.get("CHATBOT_TOP_K", "15")),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        shop_policies_path=Path(os.environ.get("SHOP_POLICIES_FILE", "generate_prompt/shop_policies.json")),
    )


def main() -> int:
    """Local dev-server entry point: python -m connect_whatsapp.whatsapp_server.
    Runs Flask's own dev server (fine for local testing against an ngrok/
    Cloudflare tunnel) -- a real deployment should use make_app() under a
    proper WSGI server instead (see connect_whatsapp/whatsapp-integration.md)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menu", type=Path, default=Path("structure.json"))
    parser.add_argument("--index", default=os.environ.get("PINECONE_INDEX", "coffee-menu"))
    parser.add_argument("--namespace", default=os.environ.get("PINECONE_NAMESPACE", ""))
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    # Lives in generate_prompt/, not this file's own directory -- shared
    # with the terminal chatbot, not duplicated per interface. Resolved
    # relative to the working directory, matching --menu's default above;
    # run_whatsapp_server.ps1 always launches from the project root.
    parser.add_argument("--shop-policies", type=Path,
                         default=Path("generate_prompt/shop_policies.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    try:
        app = _build_flask_app(
            menu_path=args.menu, index_name=args.index, namespace=args.namespace or None,
            top_k=args.top_k, gemini_model=args.gemini_model, shop_policies_path=args.shop_policies,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    logger.info("Listening on %s:%d.", args.host, args.port)
    # threaded=True so one customer's slow Gemini/Pinecone round trip
    # doesn't block every other customer's webhook call behind it. Flask's
    # own dev server is still fine for a demo, but a real deployment should
    # run this app under a proper WSGI server instead (gunicorn/waitress) --
    # see connect_whatsapp/whatsapp-integration.md.
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
