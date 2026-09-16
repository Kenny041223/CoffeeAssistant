#!/usr/bin/env bash
# Render Start Command for a single Web Service running BOTH processes this
# architecture needs: the webhook (accepts/queues messages) and the worker
# (generates and sends replies). They coordinate through a local SQLite file
# (see connect_whatsapp/message_store.py's module docstring for why that file
# must live on this same container's disk, never a network filesystem) --
# running both in one container is what lets them share it without needing
# two separate Render services or a network database.
#
# Requires a Render "Disk" attached to this service (see
# connect_whatsapp/whatsapp-integration.md) so WHATSAPP_DB_PATH survives
# restarts/redeploys instead of resetting to an empty queue each time.
#
# If either process exits for any reason, this script stops the other and
# exits non-zero, so Render's restart-on-failure policy restarts the whole
# service cleanly -- rather than silently running in a half-working state
# (e.g. the webhook still accepting messages while no worker processes them).
set -uo pipefail

python -m connect_whatsapp.message_worker &
worker_pid=$!

gunicorn --bind "0.0.0.0:${PORT:-8000}" 'connect_whatsapp.whatsapp_server:make_app()' &
web_pid=$!

# `wait -n` returns the exit status of whichever job finishes first. Capture
# it via `||`, not `if ! wait -n ...; then exit_code=$?; fi` -- under `set -e`
# semantics the latter reports the if-test's own status, not the failing
# child's actual code (verified by hand; it silently produced 0 instead of
# the real exit code). No `set -e` here for the same reason: it would abort
# the script the instant either child (correctly) returns non-zero, skipping
# the cleanup below.
exit_code=0
wait -n "$worker_pid" "$web_pid" || exit_code=$?
kill "$worker_pid" "$web_pid" 2>/dev/null || true
wait "$worker_pid" "$web_pid" 2>/dev/null || true
exit "$exit_code"
