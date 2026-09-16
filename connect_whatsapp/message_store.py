"""Durable inbox, outgoing replies and sessions for a single host with local disk.

SQLite transactions coordinate independent web/worker processes. Never place this
database on a network filesystem. Lease tokens fence stale workers' DB writes;
the remote send itself cannot participate in the database transaction.
"""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SESSION_TTL = 24 * 60 * 60
DEDUP_RETENTION = 8 * 24 * 60 * 60
FAILED_RETENTION = 30 * 24 * 60 * 60
LEASE_SECONDS = 90
MAX_ATTEMPTS = 5


class LeaseLost(RuntimeError):
    pass


@dataclass(frozen=True)
class IncomingMessage:
    message_id: str
    sender: str
    text: str
    timestamp: float


@dataclass(frozen=True)
class Job:
    message_id: str
    sender: str
    text: str
    timestamp: float
    token: str
    attempts: int
    reply_text: str | None


class MessageStore:
    def __init__(self, path: Path, *, clock=time.time):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        with self._db() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError("Unsupported WhatsApp database schema")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL UNIQUE,
                    sender TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    received_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'processing', 'sent', 'failed')),
                    available_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    token TEXT,
                    lease_until REAL,
                    reply_text TEXT,
                    next_state TEXT,
                    error TEXT,
                    finished_at REAL
                );
                CREATE INDEX IF NOT EXISTS messages_sender ON messages(sender, seq, status);
                CREATE INDEX IF NOT EXISTS messages_queue ON messages(status, available_at);
                CREATE TABLE IF NOT EXISTS sessions (
                    sender TEXT PRIMARY KEY, state TEXT NOT NULL, expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY, heartbeat REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                PRAGMA user_version=1;
            """)

    @contextmanager
    def _db(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    def bind_phone_number(self, phone_number_id: str) -> None:
        """Prevent a misconfigured worker from sending another shop's queued replies."""
        with self._db(write=True) as db:
            db.execute("INSERT INTO settings VALUES ('phone_number_id', ?) ON CONFLICT(name) DO NOTHING",
                       (phone_number_id,))
            actual = db.execute("SELECT value FROM settings WHERE name='phone_number_id'").fetchone()[0]
            if actual != phone_number_id:
                raise ValueError("Database belongs to a different WhatsApp phone number ID")

    def enqueue(self, messages: list[IncomingMessage]) -> int:
        """Commit the entire batch before HTTP acknowledgment; duplicates are no-ops."""
        now = self.clock()
        with self._db(write=True) as db:
            count = 0
            for message in messages:
                result = db.execute("""
                    INSERT INTO messages(message_id, sender, text, timestamp, received_at, available_at)
                    VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(message_id) DO NOTHING
                """, (message.message_id, message.sender, message.text, message.timestamp, now, now))
                count += result.rowcount
            return count

    def claim(self) -> Job | None:
        now = self.clock()
        with self._db(write=True) as db:
            row = db.execute("""
                SELECT m.* FROM messages m
                WHERE ((m.status='pending' AND m.available_at<=?)
                    OR (m.status='processing' AND m.lease_until<=?))
                AND NOT EXISTS (
                    SELECT 1 FROM messages earlier
                    WHERE earlier.sender=m.sender AND earlier.seq<m.seq
                    AND earlier.status IN ('pending', 'processing')
                ) ORDER BY m.seq LIMIT 1
            """, (now, now)).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            db.execute("""UPDATE messages SET status='processing', token=?, lease_until=?,
                       attempts=attempts+1 WHERE message_id=?""",
                       (token, now + LEASE_SECONDS, row["message_id"]))
            return Job(row["message_id"], row["sender"], row["text"], row["timestamp"],
                       token, row["attempts"] + 1, row["reply_text"])

    def _owned(self, db, job: Job):
        row = db.execute("""SELECT * FROM messages WHERE message_id=? AND token=?
                         AND status='processing' AND lease_until>?""",
                         (job.message_id, job.token, self.clock())).fetchone()
        if row is None:
            raise LeaseLost("Message lease is no longer owned by this worker")
        return row

    def renew(self, job: Job) -> None:
        with self._db(write=True) as db:
            self._owned(db, job)
            db.execute("UPDATE messages SET lease_until=? WHERE message_id=?",
                       (self.clock() + LEASE_SECONDS, job.message_id))

    def save_reply(self, job: Job, text: str, state: dict) -> None:
        with self._db(write=True) as db:
            self._owned(db, job)
            db.execute("UPDATE messages SET reply_text=?, next_state=? WHERE message_id=?",
                       (text, json.dumps(state), job.message_id))

    def complete(self, job: Job) -> None:
        with self._db(write=True) as db:
            row = self._owned(db, job)
            if not row["reply_text"] or not row["next_state"]:
                raise ValueError("A reply must be persisted before completion")
            now = self.clock()
            db.execute("""INSERT INTO sessions(sender, state, expires_at) VALUES (?, ?, ?)
                       ON CONFLICT(sender) DO UPDATE SET state=excluded.state, expires_at=excluded.expires_at""",
                       (job.sender, row["next_state"], now + SESSION_TTL))
            # Retain the ID tombstone for deduplication without retaining the full message.
            db.execute("""UPDATE messages SET status='sent', finished_at=?, token=NULL,
                       lease_until=NULL, text='', reply_text=NULL, next_state=NULL, error=NULL
                       WHERE message_id=?""", (now, job.message_id))

    def fail(self, job: Job, error: str, *, retryable=True) -> None:
        with self._db(write=True) as db:
            self._owned(db, job)
            terminal = not retryable or job.attempts >= MAX_ATTEMPTS
            now = self.clock()
            db.execute("""UPDATE messages SET status=?, error=?, available_at=?, token=NULL,
                       lease_until=NULL, finished_at=? WHERE message_id=?""",
                       ("failed" if terminal else "pending", error[:200],
                        now + min(300, 2 ** min(job.attempts, 9)),
                        now if terminal else None, job.message_id))

    def session(self, sender: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT state FROM sessions WHERE sender=? AND expires_at>?",
                             (sender, self.clock())).fetchone()
        return json.loads(row["state"]) if row else {}

    def heartbeat(self, worker_id: str) -> None:
        with self._db(write=True) as db:
            db.execute("""INSERT INTO workers VALUES (?, ?) ON CONFLICT(worker_id)
                       DO UPDATE SET heartbeat=excluded.heartbeat""", (worker_id, self.clock()))

    def remove_worker(self, worker_id: str) -> None:
        with self._db(write=True) as db:
            db.execute("DELETE FROM workers WHERE worker_id=?", (worker_id,))

    def stats(self) -> dict:
        with self._db() as db:
            counts = {r["status"]: r["n"] for r in db.execute(
                "SELECT status, COUNT(*) AS n FROM messages GROUP BY status")}
            workers = db.execute("SELECT COUNT(*) FROM workers WHERE heartbeat>?",
                                 (self.clock() - LEASE_SECONDS,)).fetchone()[0]
        return {"pending": counts.get("pending", 0), "processing": counts.get("processing", 0),
                "failed": counts.get("failed", 0), "workers": workers}

    def failed_messages(self) -> list[dict]:
        with self._db() as db:
            return [dict(row) for row in db.execute(
                "SELECT message_id, attempts, error, finished_at FROM messages WHERE status='failed' ORDER BY seq LIMIT 100")]

    def retry_failed(self, message_id: str) -> None:
        with self._db(write=True) as db:
            row = db.execute("SELECT * FROM messages WHERE message_id=? AND status='failed'",
                             (message_id,)).fetchone()
            if row is None:
                raise ValueError("No failed message with that ID")
            if row["timestamp"] < self.clock() - SESSION_TTL:
                raise ValueError("Customer service window has expired; ask the customer to message again")
            if db.execute("SELECT 1 FROM messages WHERE sender=? AND seq>? LIMIT 1",
                          (row["sender"], row["seq"])).fetchone():
                raise ValueError("A newer message exists; do not replay stale conversation state")
            db.execute("""UPDATE messages SET status='pending', attempts=0, available_at=?,
                       finished_at=NULL, error=NULL WHERE message_id=?""", (self.clock(), message_id))

    def cleanup(self) -> None:
        now = self.clock()
        with self._db(write=True) as db:
            db.execute("DELETE FROM sessions WHERE expires_at<=?", (now,))
            db.execute("DELETE FROM workers WHERE heartbeat<?", (now - SESSION_TTL,))
            db.execute("DELETE FROM messages WHERE status='sent' AND finished_at<?", (now - DEDUP_RETENTION,))
            db.execute("DELETE FROM messages WHERE status='failed' AND finished_at<?", (now - FAILED_RETENTION,))
