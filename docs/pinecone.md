# Pinecone vector sync

This stage reads the existing `data/menu-embeddings.json` vectors (produced by
the embedding stage, see [embeddings.md](embeddings.md)) plus `structure.json`
for metadata, and upserts each product as one vector into a
[Pinecone](https://www.pinecone.io/) index. `structure.json` stays the sole
menu dataset; Pinecone only ever gets a copy of what's needed to filter and
display a similarity match (name, category, aliases, price summary) alongside
the vector, keyed by `product_id`.

**Implementation status:** this stage has been set up but never run. No
Pinecone account, index, or API key has been used with this code yet.

## One-time account setup (you do this, not this tool)

1. Create a Pinecone account at [pinecone.io](https://www.pinecone.io/) if you
   don't have one, and generate an API key from the Pinecone console.
2. Set it as an environment variable in your shell — never pass it as a
   command-line argument, and never commit it:
   ```powershell
   $env:PINECONE_API_KEY = "your-key-here"
   ```
   This only lasts for the current shell session. For a persistent value, set
   it as a Windows user environment variable instead, or put it in a local
   `.env` file (already covered by `.gitignore`) and load it into the shell
   yourself before running the script.

## Install (main `.venv`, no GPU needed)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-pinecone.txt
```

This only adds the `pinecone` client package on top of the existing
`requirements.txt`. It does not need `.venv-embeddings` or a GPU — syncing
already-computed vectors is plain HTTP.

## Preview before touching your Pinecone account

```powershell
.\scripts\push_pinecone.ps1 -DryRun
```

`-DryRun` makes **zero network calls** — it loads `menu-embeddings.json` and
`structure.json` locally, verifies they still match each other (see
Consistency checks below), and prints exactly what would be upserted: each
product's ID, name, category, and price summary. Nothing is created,
modified, or deleted in Pinecone. Use this to check the plan is what you
expect before running for real.

## Run for real

```powershell
# Defaults: index "coffee-menu", default namespace, aws/us-east-1 serverless
.\scripts\push_pinecone.ps1

# Custom index/namespace/region
.\scripts\push_pinecone.ps1 -IndexName my-menu -Namespace staging -Cloud gcp -Region us-central1
```

If the named index doesn't exist yet, it is created as a serverless index
with dimension 1024 and cosine metric, matching the embedding stage's output
exactly. If it already exists, its dimension and metric are assumed to
already match — the script does not attempt to change an existing index's
configuration.

## Consistency checks

Before any upsert, the sync refuses to run unless:
- `menu-embeddings.json`'s recorded `menu_sha256` matches the actual current
  `structure.json` (protects against syncing vectors from a stale or edited
  menu version).
- Every embedded product's `text_sha256` still matches that product's current
  `search_text` (protects against a product being hand-edited without
  regenerating its embedding).
- Every embedded `product_id` still exists in the current menu.

Any mismatch stops the whole sync with a clear error rather than upserting
partially-stale data.

## Two-way sync

A product removed from the menu since the last sync is deleted from the
Pinecone index, not just left behind as an orphaned vector — the script lists
the index's existing vector IDs, computes the difference against the current
menu's product IDs, and deletes anything no longer present. (This comparison
needs a live index, so it does not run under `-DryRun`.)

## What's still future work

Querying (`encode_query()` already exists in
[`embed_menu.py`](../app/services/embed_menu.py) for later use), a retrieval
service, and the customer-facing assistant are not part of this stage. This
stage only keeps Pinecone's contents in sync with the local menu.
