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
    -> Qwen3-Embedding encodes one search_text per product
    -> Save product IDs and vectors in generate_embedding/data/menu-embeddings.json
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

Follow the [embedding setup guide](generate_embedding/embeddings.md) to create
a separate environment with Pascal-compatible PyTorch and cache the Qwen model
once on the GTX 1070 Ti PC. For normal runs:

```powershell
.\generate_embedding\generate_embeddings.ps1
```

This uses Hugging Face `Qwen/Qwen3-Embedding-0.6B` at a fixed revision, FP32,
standard attention, a batch size of two, and all 1,024 dimensions. The output
contains vectors and references to `structure.json`; it does not create another
menu catalog. The script always uses cached model files and fails if the cache
is incomplete. It does not install dependencies or download weights.

Vectors sync to a [Pinecone](https://www.pinecone.io/) index with
[`generate_embedding/push_pinecone.ps1`](generate_embedding/pinecone.md) --
see that guide for account setup, the `-DryRun` preview, and the two-way sync
that removes vectors for products no longer in the menu. A retrieval endpoint
and the customer-facing chatbot have not been implemented yet.

## Project layout

Everything the pipeline runs -- schemas, OCR, both menu-generation
pipelines, embeddings, and Pinecone sync -- lives in one folder:

```text
generate_embedding/menu.py                  Final menu schema (shared by every stage)
generate_embedding/qwen_ocr.py              OCR transcription schema
generate_embedding/ocr.py                   Image transcription through Ollama (shared)
generate_embedding/structure_menu.py        Batch menu generation, validation, saving
generate_embedding/structure_menu_v2.py     Staged menu generation (survey + per-item detail)
generate_embedding/generate_structureFile.ps1 Run OCR and menu generation
generate_embedding/menu-structure.md        Menu generation docs
generate_embedding/embed_menu.py            Qwen document and query embeddings
generate_embedding/embeddings.py            Embedding vector schema
generate_embedding/pinecone_sync.py         Sync vectors to a Pinecone index
generate_embedding/generate_embeddings.ps1  Run embeddings from the cached model
generate_embedding/push_pinecone.ps1        Sync embeddings to Pinecone
generate_embedding/requirements-embeddings.txt Optional deps for the embedding PC
generate_embedding/requirements-pinecone.txt   Optional deps for the Pinecone sync
generate_embedding/embeddings.md            Embedding stage docs
generate_embedding/pinecone.md              Pinecone sync docs
generate_embedding/scripts/start-qwen.ps1 Portable Ollama setup helper (shared)
generate_embedding/tests/    Automated tests with mock inference
docs/                        OCR setup and customer-assistant design notes
generate_embedding/image/    Input menu images
generate_embedding/data/qwen-ocr/     Local OCR evidence, generated at runtime
generate_embedding/data/structure-runs/ Local prompts, responses, and run manifests
generate_embedding/data/menu-embeddings.json Derived vectors, created by the embedding stage
structure.json               Final menu, created after a successful model run
```

## Next milestones

Done: menu generation (both pipelines), human review against source images,
embedding generation, and Pinecone sync -- all run for real, not just coded.

- Expose a retrieval endpoint (FastAPI) over the Pinecone index plus
  structured price/size/temperature filtering (semantic similarity alone
  doesn't enforce numeric constraints).
- Evaluate expected product matches and duplicate handling with a real query
  set -- see [`customer-assistant-notes.md`](docs/customer-assistant-notes.md)
  for captured behavior requirements (e.g. the house-blend/seasonal bean flow).
- Build the customer assistant, controlled tool calls, and answer-quality
  evaluation.
- Operationalize the re-run cycle for when the menu changes (seasonal batch
  rotation, price updates): re-run OCR -> structure generation -> embeddings
  -> Pinecone sync; the consistency checks already refuse stale data.

The API, customer assistant, and deployment are still planned work.
