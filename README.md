# Coffee Shop AI Customer Service Agent

A portfolio project that turns coffee shop menu images into structured data for
a future customer service assistant. The current implementation focuses on local
Qwen inference, structured output validation, traceable extraction, and product
embeddings for later semantic search.

## Current implementation

```text
Menu images
    -> Qwen3-VL transcribes each image
    -> Qwen text model reads the complete OCR batch
       and generates one consolidated menu
    -> Python validates, attaches provenance, and saves structure.json
    -> Gemini (hosted) encodes one search_text per product
    -> Save product IDs and vectors in generate_embedding/data/menu-embeddings.json
    -> Synced to Pinecone; a Gemini-powered chatbot retrieves and answers from it
```

**Qwen generates the menu content:** product identities, descriptions, variants,
aliases, source evidence, uncertainty notes, and `search_text`. It sees the OCR
from all images together so a price-list entry and a description poster can
contribute to the same product. Python supplies the schema and checks the
response, adds IDs and run metadata, and writes the final file. It does not
merge products or fill in menu facts after generation.

`structure.json` is the only final menu dataset. OCR files and generation logs
are retained as evidence. There is no separate menu catalog or manual correction
file in the active pipeline.

**Validation status:** OCR and menu-generation unit tests use mock model
responses. The full pipeline has since been run end-to-end against real menu
photos with real models: menu generation, human review against the source
images (see `structure.json`'s per-product `issues` fields for what was
manually corrected and why), embedding generation on the GTX 1070 Ti PC, and
a Pinecone sync -- all independently verified. Semantic retrieval *quality*
(are the right products actually returned for real customer queries) still
requires evaluation; nothing here claims that yet.

## First-time setup on the model-capable PC

Install Python 3.12 and set up Ollama using [the setup guide](docs/ocr.md). From
the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Normal Ollama installation: download both models once
ollama pull qwen3-vl:4b-instruct
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

For portable Ollama setup, follow the separate instructions in the setup guide.

## Generate the menu

Start Ollama first. The run script uses the existing environment and installed
models; it does not install packages, download models, or start the server.
Its default server is `http://127.0.0.1:11434`. The models are
`qwen3-vl:4b-instruct` for OCR and `qwen3:4b-instruct-2507-q4_K_M` for menu
generation. The script stops if either pipeline stage fails.

```powershell
# Images -> OCR -> model-generated structure.json
.\generate_embedding\generate_structureFile.ps1

# Reuse existing OCR files and run only menu generation
.\generate_embedding\generate_structureFile.ps1 -SkipOcr

# Portable Ollama server using the project's separate port
.\generate_embedding\generate_structureFile.ps1 -OllamaUrl http://127.0.0.1:11435

# Supply currency only when confirmed from the menu or shop
.\generate_embedding\generate_structureFile.ps1 -SkipOcr -Currency MYR

# This laptop: inspect the real prompt/schema/input without loading any model
# Requires existing OCR JSON in generate_embedding/data/qwen-ocr/
.\generate_embedding\generate_structureFile.ps1 -PrepareOnly

# Mock inference tests; no model download or server required
.\.venv\Scripts\python.exe -m unittest discover -s generate_embedding/tests -v
```

`-PrepareOnly` skips OCR, saves the actual generation request artifacts, and
does not create `structure.json`. See [menu generation](generate_embedding/menu-structure.md)
for schema details, validation, run evidence, and memory-related options. A
second, staged pipeline (`generate_embedding/structure_menu_v2.py`) also exists --
see its module docstring for why it's now the primary path.

## Why one record per product

Repeated menu photos should contribute evidence to a shared product instead of
creating repeated search results. The Qwen prompt asks for one record per
product, retaining separate identities for distinct blends or similarly named
drinks. Python rejects detectable duplicate identities and invalid references,
then asks Qwen to correct its response once. This catches structural problems;
it does not prove that every model merge is factually correct.

Each product contains model-generated `search_text` alongside exact price
variants and source references. The optional embedding stage encodes that text
once per product and retains its ID for later structured lookup. Numeric price
constraints need structured filters.

## Product embeddings and vector search

Follow the [embedding setup guide](generate_embedding/embeddings.md) -- no
GPU needed, it's a hosted API call. For normal runs:

```powershell
.\generate_embedding\generate_embeddings.ps1
```

This uses Google's `gemini-embedding-001` via the hosted Gemini API, 768
dimensions (truncated from the native 3072, re-normalized client-side), and
a batch size of ten. The output contains vectors and references to
`structure.json`; it does not create another menu catalog.

Vectors sync to a [Pinecone](https://www.pinecone.io/) index with
[`generate_embedding/push_pinecone.ps1`](generate_embedding/pinecone.md) --
see that guide for account setup, the `-DryRun` preview, and the two-way sync
that removes vectors for products no longer in the menu.

## Customer chatbot

```powershell
.\generate_prompt\run_chatbot.ps1
```

An interactive, terminal-based chat loop: embeds the customer's message
(same Gemini embedding model as the menu, so retrieval actually works),
searches Pinecone, pulls each match's full record from `structure.json`
(Pinecone's metadata is a flattened summary, not the source of truth), and
hands that as grounding context to Gemini for the actual reply. Gemini's own
chat session carries conversation history; retrieval context is refreshed
every turn. See [generate_prompt/chatbot.py](generate_prompt/chatbot.py)'s
module docstring and `SYSTEM_PROMPT` for the grounding rules (never invent a
price, ask house-blend-vs-seasonal before quoting one, stay on menu topics).
Nothing here runs locally -- retrieval and generation are both hosted.

Shop-level facts that aren't about any one product -- opening hours, whether
customers can bring their own coffee beans -- come from
[generate_prompt/shop_policies.json](generate_prompt/shop_policies.json)
(loaded by [generate_prompt/shop_policies.py](generate_prompt/shop_policies.py))
rather than `structure.json`; see
[docs/customer-assistant-notes.md](docs/customer-assistant-notes.md) for why.

## WhatsApp

```powershell
.\connect_whatsapp\run_whatsapp_server.ps1
```

Bridges the same chatbot core to a real WhatsApp number via Meta's official
WhatsApp Cloud API -- a webhook server, not a terminal loop, so it needs a
Meta Business/Developer app and a public HTTPS URL Meta can reach. No reply
logic is duplicated: [connect_whatsapp/whatsapp_server.py](connect_whatsapp/whatsapp_server.py)
only handles receiving Meta's webhook calls and sending replies back
through the Graph API, calling straight into `generate_prompt/chatbot.py`'s
`build_engine()`/`reply()`, with one conversation kept per customer phone
number. Full setup walkthrough (Meta app config, credentials, webhook,
local testing with a tunnel vs. real deployment) in
[connect_whatsapp/whatsapp-integration.md](connect_whatsapp/whatsapp-integration.md).

## Project layout

Everything the pipeline runs -- schemas, OCR, both menu-generation
pipelines, embeddings, and Pinecone sync -- lives in one folder, plus the
chatbot in its own, and the WhatsApp bridge in a third:

```text
generate_embedding/menu.py                  Final menu schema (shared by every stage)
generate_embedding/qwen_ocr.py              OCR transcription schema
generate_embedding/ocr.py                   Image transcription through Ollama (shared)
generate_embedding/structure_menu.py        Batch menu generation, validation, saving
generate_embedding/structure_menu_v2.py     Staged menu generation (survey + per-item detail)
generate_embedding/generate_structureFile.ps1 Run OCR and menu generation
generate_embedding/menu-structure.md        Menu generation docs
generate_embedding/embed_menu.py            Gemini document and query embeddings
generate_embedding/embeddings.py            Embedding vector schema
generate_embedding/pinecone_sync.py         Sync vectors to a Pinecone index
generate_embedding/generate_embeddings.ps1  Run embeddings via the Gemini API
generate_embedding/push_pinecone.ps1        Sync embeddings to Pinecone
generate_embedding/requirements-gemini.txt     Deps for Gemini embedding/generation (no GPU)
generate_embedding/requirements-pinecone.txt   Deps for the Pinecone sync
generate_embedding/embeddings.md            Embedding stage docs
generate_embedding/pinecone.md              Pinecone sync docs
generate_embedding/scripts/start-qwen.ps1 Portable Ollama setup helper (shared)
generate_embedding/tests/    Automated tests with mock inference
generate_prompt/chatbot.py              Customer chat assistant (retrieval + Gemini reply)
generate_prompt/shop_policies.py        Shop-level facts schema/loader (not menu data)
generate_prompt/shop_policies.json      Shop policy facts (hours, BYO beans, etc.)
generate_prompt/run_chatbot.ps1         Run the chatbot
generate_prompt/requirements-chatbot.txt Deps for the chatbot (no GPU)
generate_prompt/tests/                  Automated tests with mock inference
connect_whatsapp/whatsapp_server.py     WhatsApp Cloud API webhook bridge (reuses chatbot.py)
connect_whatsapp/run_whatsapp_server.ps1 Run the WhatsApp webhook server
connect_whatsapp/requirements-whatsapp.txt Deps for the WhatsApp server
connect_whatsapp/whatsapp-integration.md Meta setup and deployment walkthrough
connect_whatsapp/tests/                 Automated tests with mock Meta/Gemini/Pinecone calls
docs/                        OCR setup and customer-assistant design notes
generate_embedding/image/    Input menu images
generate_embedding/data/qwen-ocr/     Local OCR evidence, generated at runtime
generate_embedding/data/structure-runs/ Local prompts, responses, and run manifests
generate_embedding/data/menu-embeddings.json Derived vectors, created by the embedding stage
structure.json               Final menu, created after a successful model run
```

## Next milestones

Done: menu generation (both pipelines), human review against source images,
embedding generation, Pinecone sync, and a working terminal chatbot
(retrieval + Gemini reply generation) -- all run for real, not just coded.

- Structured price/size/temperature filtering alongside semantic search
  (similarity alone doesn't enforce numeric constraints -- e.g. "under 15").
- Broader conversational evaluation: expected product matches, duplicate
  handling, and multi-turn behavior with a real query set -- see
  [`customer-assistant-notes.md`](docs/customer-assistant-notes.md) for
  captured behavior requirements.
- An actual ordering flow (confirm an order, not just answer questions) and
  controlled tool calls, rather than a pure Q&A chat loop.
- A non-terminal interface (web/API) in front of `generate_prompt/chatbot.py`.
- Operationalize the re-run cycle for when the menu changes (seasonal batch
  rotation, price updates): re-run OCR -> structure generation -> embeddings
  -> Pinecone sync; the consistency checks already refuse stale data.

Deployment and a non-terminal interface are still planned work.
