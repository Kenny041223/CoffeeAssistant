# Gemini product embeddings

This stage reads the existing `structure.json` and embeds each product's
`search_text` once using Google's
[`gemini-embedding-001`](https://ai.google.dev/gemini-api/docs/embeddings)
via the hosted Gemini API. Output is truncated (Matryoshka Representation
Learning) from the model's native 3072 dimensions down to 768, then
normalized client-side -- Google's own docs note that only the untruncated
output is pre-normalized, so this stage always re-normalizes regardless of
requested dimension. Menu generation continues to use Ollama locally; this
stage and the chatbot both use the hosted Gemini API instead.

**Implementation status:** run successfully -- 21 product vectors generated
in ~4.4s, independently verified (schema, menu-hash match, normalized
768-dim vectors) before being synced to Pinecone.

## Setup

No GPU, no PyTorch, no model download or caching -- embedding is a plain API
call. Install into the main `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r generate_embedding\requirements-gemini.txt
```

You need a free Gemini API key (see the main [README](../README.md) or
[pinecone.md](pinecone.md) for the `.env` setup pattern) -- the same
`GEMINI_API_KEY` used for the chatbot's reply generation covers this stage
too. Nothing extra to register.

## Normal runs

```powershell
.\generate_embedding\generate_embeddings.ps1

# Custom menu and batch size
.\generate_embedding\generate_embeddings.ps1 -InputFile structure.json -Output generate_embedding/data/menu-embeddings.json -BatchSize 10
```

The script loads `GEMINI_API_KEY` from `.env` automatically (see
`generate_embeddings.ps1`'s own `.env` loader). The default input is
`structure.json` and output is `generate_embedding/data/menu-embeddings.json`.

| Setting | Default | Purpose |
|---|---|---|
| `-BatchSize` | `10` | Products per `embed_content` API call |
| `-MaxLength` | `2048` | Rough character-based cap per product's `search_text`, checked before any API call (no local tokenizer for a hosted model, so this is a conservative chars-per-token proxy, not an exact count) |
| Dimensions | `768` | Truncated from the native 3072; re-normalized client-side |
| Similarity | cosine | Matches the Pinecone index's configured metric |

An input longer than `-MaxLength` (by the character proxy) is rejected
instead of silently truncating menu details -- review that product's
`search_text`, or raise the limit deliberately if it's a legitimately long
description.

## Output and later retrieval

`structure.json` stays the only menu dataset. `generate_embedding/data/menu-embeddings.json`
is a derived vector artifact containing product IDs, vectors, text hashes,
the input menu hash, and embedding configuration. It references the menu
instead of duplicating product descriptions, price variants, or OCR
evidence. Only products are embedded at this stage; series and add-ons
remain in the canonical menu.

The stage validates the menu and vector output and publishes the vector
file atomically after success. A failed run leaves any previous vector file
intact. Regenerate the artifact after changing the menu and use its hashes
to prevent mixing vectors with a different menu version -- `pinecone_sync.py`
refuses to sync if these hashes don't match the current `structure.json`.

`GeminiEmbedder` in `generate_embedding/embed_menu.py` exposes
`encode_documents()` and `encode_query()` -- the same interface the chatbot
uses for query-time retrieval, so both paths stay on the exact same model,
dimensions, and normalization.

A vector database now stores these vectors under `product_id` for retrieval
-- see [pinecone.md](pinecone.md). Apply price, size, and temperature
constraints to structured variants; semantic similarity alone does not
enforce them. See [../generate_prompt/](../generate_prompt/) for the actual
customer-facing chatbot built on top of this.
