# Coffee Shop AI Customer Service Agent

A portfolio project that turns coffee shop menu images into structured data for
a future customer service assistant. The current implementation focuses on local
Qwen inference, structured output validation, and traceable extraction.

## Current implementation

```text
Menu images
    -> Qwen3-VL transcribes each image
    -> Qwen text model reads the complete OCR batch
       and generates one consolidated menu
    -> Python validates, attaches provenance, and saves structure.json
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

**Validation status:** the revised pipeline is prepared for a model-capable PC.
Its automated tests use mock model responses; this revision has not yet been
run end to end with Qwen. No generated menu is shipped as proof of such a run.
Extraction accuracy and semantic retrieval quality are still unmeasured.

## Run on the model-capable PC

Install Python 3.12 and set up Ollama using [the setup guide](docs/ocr.md). From
the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Portable Ollama setup: start the server and download both models once
.\scripts\start-qwen.ps1 -PullModel -PullTextModel

# Images -> OCR -> model-generated structure.json
.\scripts\run-menu-pipeline.ps1
```

The defaults are `qwen3-vl:4b-instruct` for OCR and
`qwen3:4b-instruct-2507-q4_K_M` for menu generation. The wrapper expects Ollama
to be running and stops if either pipeline stage fails.

```powershell
# Reuse existing OCR files and run only menu generation
.\scripts\run-menu-pipeline.ps1 -SkipOcr

# Existing Ollama installation using its usual port
.\scripts\run-menu-pipeline.ps1 -OllamaUrl http://127.0.0.1:11434

# Supply currency only when confirmed from the menu or shop
.\scripts\run-menu-pipeline.ps1 -SkipOcr -Currency MYR

# This laptop: inspect the real prompt/schema/input without loading any model
# Requires existing OCR JSON in data/qwen-ocr/
.\scripts\run-menu-pipeline.ps1 -PrepareOnly

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

Each product contains model-generated `search_text` for a future embedding,
alongside exact price variants and source references. A later retrieval layer
can embed that text once per product and use its ID to retrieve structured
prices. Numeric price constraints need structured filters. No embeddings or
vector database are created by this pipeline.

## Project layout

```text
app/models/                  OCR and final menu schemas
app/services/ocr.py           Image transcription through Ollama
app/services/structure_menu.py Batch menu generation, validation, and saving
scripts/start-qwen.ps1        Portable Ollama setup helper
scripts/run-menu-pipeline.ps1 Pipeline entry point
tests/                       Automated tests with mock inference
docs/                        Setup and extraction workflow
image/                       Input menu images
data/qwen-ocr/                Local OCR evidence, generated at runtime
data/structure-runs/          Local prompts, responses, and run manifests
structure.json               Final menu, created after a successful model run
```

## Next milestones

- Run the revised pipeline on the model-capable PC and inspect its saved evidence.
- Review generated names, price mappings, identity merges, and unresolved issues
  against the original images.
- Evaluate expected product matches and duplicate handling with a small query set.
- Add an embedding index and structured menu lookup, then expose a FastAPI API.
- Build the customer assistant, controlled tool calls, and answer-quality evaluation.

The API, customer assistant, vector database, and deployment are planned work.
