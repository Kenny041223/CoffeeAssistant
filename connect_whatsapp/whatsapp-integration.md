# WhatsApp deployment and operations

The webhook validates Meta's signature, parses all text messages in the batch,
and commits them to SQLite before returning HTTP 200. A separate worker retrieves
menu facts, generates a reply, saves it, and sends it through the Graph API.

```text
Meta -> HTTPS webhook -> durable inbox -> worker -> saved reply -> Graph API
                              |             |
                        message IDs     bounded sessions
```

## Supported deployment

This implementation is intended for a low-volume shop on **one host with a local
persistent disk**. Multiple web and worker processes may share that database;
transactions and renewable leases serialize each customer's messages. Different
customers can be handled by different worker processes.

Do not put the database on NFS/SMB or deploy web and worker to separate machines
with independent disks. [SQLite WAL requires processes on the same host](https://www.sqlite.org/wal.html).
For replicas on separate hosts, replace this store with a shared transactional
database/queue before scaling. An ephemeral/free web-service filesystem is not
a durable deployment for this design.

## Configuration

Use an existing Meta WhatsApp Business app and registered business number. Set up
the app, messaging access token, and webhook subscription using the
[official Cloud API documentation](https://developers.facebook.com/docs/whatsapp/cloud-api/).
Subscribe the callback URL `https://your-domain/webhook` to the `messages` field.
Set its verify token to the same value as `WHATSAPP_VERIFY_TOKEN`.

Copy the keys from [`.env.example`](../.env.example) into local `.env`, or supply
them through your deployment's secret/environment configuration. Never commit
real credentials. The PowerShell launchers load `.env` as data; Python and
Gunicorn entry points read process environment variables only.

| Variable | Used by | Meaning |
| --- | --- | --- |
| `WHATSAPP_PHONE_NUMBER_ID` | Both | Registered business number ID; also binds the database to this number |
| `WHATSAPP_DB_PATH` | Both | Same absolute path on persistent local disk in production |
| `WHATSAPP_VERIFY_TOKEN` | Webhook | Token chosen for the subscription handshake |
| `WHATSAPP_APP_SECRET` | Webhook | Secret used to verify Meta HMAC signatures |
| `WHATSAPP_ACCESS_TOKEN` | Worker | Graph API messaging credential |
| `GEMINI_API_KEY` | Worker | Embedding and response generation |
| `PINECONE_API_KEY` | Worker | Menu retrieval |
| `WHATSAPP_GRAPH_VERSION` | Worker | Graph API version; defaults to `v21.0` |
| `MENU_FILE` | Worker | Defaults to `structure.json` |
| `PINECONE_INDEX` | Worker | Defaults to `coffee-menu` |
| `PINECONE_NAMESPACE` | Worker | Defaults to the default namespace |
| `CHATBOT_TOP_K` | Worker | Between 1 and 50; defaults to 15 |
| `GEMINI_MODEL` | Worker | Defaults to `gemini-3.5-flash-lite` |
| `SHOP_POLICIES_FILE` | Worker | Defaults to `generate_prompt/shop_policies.json` |

## Local development on Windows

Install dependencies from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r connect_whatsapp\requirements-whatsapp.txt
```

Run each command in a separate terminal:

```powershell
.\connect_whatsapp\run_whatsapp_server.ps1
.\connect_whatsapp\run_whatsapp_worker.ps1
```

The webhook listens on `127.0.0.1:8000`. Point an HTTPS development tunnel at that
port and configure its `/webhook` URL in Meta. Both launchers use
`connect_whatsapp/data/messages.sqlite3` unless `WHATSAPP_DB_PATH` overrides it.
The worker must be running for queued messages to receive replies.

## Production startup on Render

Render's standard Web Service runs one process, but this architecture needs
two (webhook + worker) sharing one local SQLite file. The practical fix on
Render: run **both processes in the same service** via
[`connect_whatsapp/start_render.sh`](start_render.sh), which launches the
worker in the background and gunicorn in the foreground, sharing that one
container's disk. If either process exits, the script stops the other and
exits non-zero, so Render's restart-on-failure policy restarts the whole
service cleanly rather than leaving it half-working (e.g. the webhook still
accepting messages with no worker to answer them).

1. **Add a Disk** to the Render service (Settings > Disks > Add Disk) -- pick
   a mount path, e.g. `/var/data`. Without this, the SQLite file (and the
   whole message queue) resets to empty on every restart/redeploy, since a
   Render service's own filesystem is otherwise ephemeral.
2. **Set `WHATSAPP_DB_PATH`** in the Environment tab to a path inside that
   mount, e.g. `/var/data/messages.sqlite3`.
3. **Build Command**: unchanged --
   `pip install -r connect_whatsapp/requirements-whatsapp.txt`.
4. **Start Command**: `bash connect_whatsapp/start_render.sh`.
5. All the other environment variables from the table above still apply.

This has been verified as a standalone supervisor pattern (equivalent dummy
processes: killing either one correctly stops the other and propagates its
exit code, with no orphaned process left running) but **not yet exercised
with the real worker and gunicorn together** -- `gunicorn` is POSIX-only and
doesn't run at all on Windows, so this gets its first real end-to-end test
on Render itself. Watch the deploy logs for both "Booting worker" (gunicorn)
and the message-worker's own startup log line after deploying, and send a
real test message before considering it verified.

## Production startup on Linux

Install the repository and virtual environment at `/opt/coffee-assistant`, create
a dedicated `coffee` service user, and store credentials in a root-owned
`/etc/coffee-assistant.env` readable only by the service manager. In that file set:

```text
WHATSAPP_DB_PATH=/var/lib/coffee-assistant/messages.sqlite3
```

The example units [coffee-web.service](deploy/coffee-web.service) and
[coffee-worker.service](deploy/coffee-worker.service) supervise the two processes,
create the private state directory, and restart failed processes. Adjust paths
and service user to the host before installing/enabling these units. They are
templates; nothing in the repository installs a service automatically.

The underlying start commands, run from the project root with environment set:

```sh
.venv/bin/gunicorn --workers 2 --bind 127.0.0.1:8000 'connect_whatsapp.whatsapp_server:make_app()'
.venv/bin/python -m connect_whatsapp.message_worker
```

Put the webhook behind an HTTPS reverse proxy. Keep the database outside the
application release directory and restrict its directory to the service user.
The same directory contains SQLite's WAL/SHM files. Back up the database through
SQLite's online backup API, or stop **both** services before copying it; copying
only the main file while it is running can omit committed WAL data.

Gunicorn calls factories using the quoted `module:function()` expression;
there is no `--factory` switch. See [Flask's deployment documentation](https://flask.palletsprojects.com/en/stable/deploying/gunicorn/).

## Delivery behavior

- Every supported message in every entry/change is processed. Other business
  numbers and unsupported media/status events are ignored. Invalid text batches
  receive 400; failed persistence receives 503, never a successful acknowledgment.
- Repeated incoming message IDs do not regenerate or resend completed replies.
  Deduplication records remain for eight days after completion.
- Only the earliest unfinished message for a customer can be claimed. Retries
  keep later messages for that customer waiting while other customers proceed.
- Claims have a 90-second lease, renewed every 20 seconds. A crashed worker's
  job can be reclaimed; stale claim tokens cannot update stored state.
- Replies are saved before sending. Network failures, HTTP 408/429 and HTTP 5xx
  are retried with exponential backoff, up to five attempts. Other Graph HTTP
  errors become failed jobs immediately. Generation failures also have a bounded
  retry budget. Conversation state advances only after Graph accepts the reply.
- **Outgoing delivery is at-least-once.** A remote acceptance followed by a lost
  response, or a crash before the completion commit, can cause a duplicate send.
  The Graph send and SQLite commit are not one transaction. Saved replies prevent
  duplicate model generation on send retries, but do not guarantee exactly-once
  delivery. `sent` means accepted by Graph, not confirmed read/delivered.
- Jobs older than 24 hours are failed without sending a free-form reply. Template
  messages and a real ordering/payment flow are not implemented.

## Monitoring and recovery

`GET /health` checks web-process liveness. `GET /ready` checks database access and
recent worker heartbeats, returning 503 if either is unavailable. It includes
pending/processing/failed counts without customer text. A live worker with failed
jobs returns 200 with `status: degraded`; alert on `failed > 0`, growing pending
counts, missing workers and low disk space. Do not use failed-job alerts to trigger
an automatic restart loop.

Inspect queue counts and up to 100 failed job IDs with sanitized errors:

```powershell
.\connect_whatsapp\run_whatsapp_worker.ps1 -Status
```

After correcting credentials or another underlying failure, retry a failed job:

```powershell
.\connect_whatsapp\run_whatsapp_worker.ps1 -RetryFailed 'wamid.example'
```

On Linux, the equivalent options are `python -m connect_whatsapp.message_worker
--status` and `--retry-failed MESSAGE_ID`. Retry is refused after the 24-hour
window or when newer messages from the customer exist; ask the customer to send
a new message in those cases. Failed jobs remain for 30 days for investigation.
Logs and the status command omit message bodies, phone numbers and credentials.

Idle sessions expire after 24 hours. Each session retains at most six complete
turns and 12,000 characters of dialogue. Retrieved menu blocks are supplied fresh
and are never stored in dialogue history. Completed job bodies are cleared,
while pending/failed jobs retain the data needed for retry. The database contains
customer data and belongs in restricted storage and backups.

## Updating menus

Generate/review the menu, regenerate embeddings, then run the sync's dry run and
sync to the intended dedicated index/namespace. Incomplete/duplicate embedding
sets and mismatched counts are rejected before any write or deletion.
For a live update, stop the worker during sync and restart it with the matching
menu file afterward; the webhook can keep queueing messages. An existing saved
outgoing reply is retained across retries, so drain/reconcile outgoing retries
before changing prices. Changed menu/policy/prompt fingerprints reset old
dialogue on the next generated reply.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s connect_whatsapp/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s generate_prompt/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s generate_embedding/tests -v
```

Tests use real temporary SQLite databases, concurrent requests/processes, and
mocked model/Graph boundaries. CI runs the suites on Windows and Linux and checks
the Gunicorn factory on Linux. A staging WhatsApp round trip is still required
to verify real credentials, provider permissions, HTTPS routing and response quality.
