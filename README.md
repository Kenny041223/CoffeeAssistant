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
    -> Save product IDs and vectors in data/menu-embeddings.json
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

**Validation status:** OCR and menu-generation tests use mock model responses.
The new embedding code and its tests have not been run, and no embedding model
or packages were installed on the laptop. The actual generated menu remains on
the model-capable PC and has not been inspected in this checkout. Extraction
accuracy and semantic retrieval quality still require evaluation.

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
.\scripts\generate_structureFile.ps1

# Reuse existing OCR files and run only menu generation
.\scripts\generate_structureFile.ps1 -SkipOcr

# Portable Ollama server using the project's separate port
.\scripts\generate_structureFile.ps1 -OllamaUrl http://127.0.0.1:11435

# Supply currency only when confirmed from the menu or shop
.\scripts\generate_structureFile.ps1 -SkipOcr -Currency MYR

# This laptop: inspect the real prompt/schema/input without loading any model
# Requires existing OCR JSON in data/qwen-ocr/
.\scripts\generate_structureFile.ps1 -PrepareOnly

# Mock inference tests; no model download or server required
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`-PrepareOnly` skips OCR, saves the actual generation request artifacts, and
does not create `structure.json`. See [menu generation](docs/menu-structure.md)
for schema details, validation, run evidence, and memory-related options.

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

## Product embeddings on the other PC

Follow the [embedding setup guide](docs/embeddings.md) to create a separate
environment with Pascal-compatible PyTorch and cache the Qwen model once on
the GTX 1070 Ti PC. For normal runs:

```powershell
.\scripts\generate_embeddings.ps1
```

This uses Hugging Face `Qwen/Qwen3-Embedding-0.6B` at a fixed revision, FP32,
standard attention, a batch size of two, and all 1,024 dimensions. The output
contains vectors and references to `structure.json`; it does not create another
menu catalog. The script always uses cached model files and fails if the cache
is incomplete. It does not install dependencies or download weights. A vector
database and chatbot have not been implemented yet.

## Project layout

```text
app/models/                  OCR and final menu schemas
app/services/ocr.py           Image transcription through Ollama
app/services/structure_menu.py Batch menu generation, validation, and saving
app/services/embed_menu.py    Optional Qwen document and query embeddings
requirements-embeddings.txt   Optional dependencies for the embedding PC
scripts/start-qwen.ps1        Portable Ollama setup helper
scripts/generate_structureFile.ps1 Run OCR and menu generation
scripts/generate_embeddings.ps1 Run embeddings from the cached model
tests/                       Automated tests with mock inference
docs/                        Setup, extraction, and embedding workflow
image/                       Input menu images
data/qwen-ocr/                Local OCR evidence, generated at runtime
data/structure-runs/          Local prompts, responses, and run manifests
data/menu-embeddings.json     Derived vectors, created by the embedding stage
structure.json               Final menu, created after a successful model run
```

## Next milestones

- Inspect the model-generated menu and saved evidence on the model-capable PC.
- Review generated names, price mappings, identity merges, and unresolved issues
  against the original images.
- Run the embedding stage on the GTX 1070 Ti PC and measure memory use.
- Add a vector database and structured menu lookup, then expose a FastAPI API.
- Evaluate expected product matches and duplicate handling with a small query set.
- Build the customer assistant, controlled tool calls, and answer-quality evaluation.

The API, customer assistant, vector database, and deployment are planned work.
