"""Run with python -m connect_whatsapp.message_worker on the webhook's host.

Generated replies are saved before sending, so delivery retries reuse them.
Graph sends are at-least-once: a timeout after remote acceptance or a crash
before the completion commit can duplicate a reply. Provider idempotency would
be necessary to guarantee exactly-once delivery.
"""
import argparse
import json
import logging
import os
import signal
import threading
import uuid
from pathlib import Path

from connect_whatsapp.message_store import LeaseLost, MAX_ATTEMPTS, MessageStore, SESSION_TTL
from connect_whatsapp.whatsapp_server import DeliveryError, require_env, send_whatsapp_message, store_from_env
from generate_prompt.chatbot import Conversation, build_engine, reply

logger = logging.getLogger(__name__)
WORKER_ENV_VARS = ("PINECONE_API_KEY", "GEMINI_API_KEY", "WHATSAPP_ACCESS_TOKEN", "WHATSAPP_PHONE_NUMBER_ID")


class MessageWorker:
    def __init__(self, store: MessageStore, engine, access_token: str, phone_number_id: str,
                 *, generate=reply, send=send_whatsapp_message):
        self.store, self.engine = store, engine
        store.bind_phone_number(phone_number_id)
        self.access_token, self.phone_number_id = access_token, phone_number_id
        self.generate, self.send = generate, send
        self.worker_id = uuid.uuid4().hex

    def process_one(self) -> bool:
        self.store.heartbeat(self.worker_id)
        job = self.store.claim()
        if job is None:
            return False
        stop = threading.Event()
        lost = threading.Event()

        def maintain_lease():
            while not stop.wait(20):
                try:
                    self.store.renew(job)
                    self.store.heartbeat(self.worker_id)
                except Exception:
                    lost.set()
                    return

        heartbeat = threading.Thread(target=maintain_lease, daemon=True)
        heartbeat.start()
        try:
            if job.attempts > MAX_ATTEMPTS or job.timestamp <= self.store.clock() - SESSION_TTL:
                self.store.fail(job, "Retry budget or customer service window expired", retryable=False)
                return True
            text = job.reply_text
            if text is None:
                conversation = Conversation.restore(self.store.session(job.sender))
                text = self.generate(self.engine, conversation, job.text)
                self.store.save_reply(job, text, conversation.snapshot())
            if lost.is_set():
                raise LeaseLost("Heartbeat failed")
            if job.timestamp <= self.store.clock() - SESSION_TTL:
                self.store.fail(job, "Customer service window expired during generation", retryable=False)
                return True
            self.store.renew(job)
            self.send(self.access_token, self.phone_number_id, job.sender, text)
            self.store.complete(job)
        except LeaseLost:
            logger.warning("Worker lost its message lease")
        except Exception as exc:
            retryable = exc.retryable if isinstance(exc, DeliveryError) else True
            # Never log/store provider response bodies, credentials or customer text.
            reason = str(exc) if isinstance(exc, DeliveryError) else type(exc).__name__
            try:
                self.store.fail(job, reason, retryable=retryable)
            except LeaseLost:
                logger.warning("Worker could not record failure after losing its lease")
            logger.warning("Message attempt failed: %s", reason)
        finally:
            stop.set()
            heartbeat.join(timeout=6)
        return True

    def run(self, stop: threading.Event) -> None:
        next_cleanup = 0.0
        try:
            while not stop.is_set():
                if self.store.clock() >= next_cleanup:
                    self.store.cleanup()
                    next_cleanup = self.store.clock() + 60
                if not self.process_one():
                    stop.wait(1)
        finally:
            self.store.remove_worker(self.worker_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument("--retry-failed", metavar="MESSAGE_ID", help="Requeue a failed message if safe to replay")
    operations.add_argument("--status", action="store_true", help="Show queue counts and up to 100 failures without customer text")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = store_from_env()
    if args.status:
        print(json.dumps({**store.stats(), "failures": store.failed_messages()}, indent=2))
        return 0
    if args.retry_failed:
        store.retry_failed(args.retry_failed)
        return 0
    env = require_env(WORKER_ENV_VARS)
    engine = build_engine(
        menu_path=Path(os.environ.get("MENU_FILE", "structure.json")),
        pinecone_key=env["PINECONE_API_KEY"], gemini_key=env["GEMINI_API_KEY"],
        index_name=os.environ.get("PINECONE_INDEX", "coffee-menu"),
        namespace=os.environ.get("PINECONE_NAMESPACE") or None,
        top_k=int(os.environ.get("CHATBOT_TOP_K", "15")),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        shop_policies_path=Path(os.environ.get("SHOP_POLICIES_FILE", "generate_prompt/shop_policies.json")),
    )
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    MessageWorker(store, engine, env["WHATSAPP_ACCESS_TOKEN"], env["WHATSAPP_PHONE_NUMBER_ID"]).run(stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
