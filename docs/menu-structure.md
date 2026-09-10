# Generate structure.json with an LLM

The pipeline uses the local Qwen3-4B-Instruct-2507 text model
(`qwen3:4b-instruct-2507-q4_K_M`). OCR still uses Qwen3-VL-4B-Instruct.
The text model needs a separate one-time download; no training or API key is needed.

The initial complete run generated a draft with 48 item records from 14 source
documents. These are model-produced records, not a count of verified unique
products. All 23 unit tests passed. Some headings/branding and add-on rows were
misclassified as products during inspection, so schema success is not content
approval. Those interpretation checks remain separate from file generation.

```text
Qwen OCR JSON -> transcription -> Qwen fills one menu schema
             -> Pydantic validation -> structure.json (draft)
```

The LLM handles interpretation of different source layouts. Python handles input
loading, validation, provenance and file writing. There are no image-specific or
menu-specific parsers. The schema is in `app/models/menu.py`; the shared extraction
prompt and batch runner are in `app/services/structure_menu.py`.

The text step uses JSON mode with a shape example, followed by Pydantic validation.
The OCR step still uses its original schema-constrained image request. The shared
prompt explains line-wrapped names and optional extras.

## Run

```powershell
.\scripts\start-qwen.ps1 -PullTextModel
.\.venv\Scripts\python.exe -m app.services.structure_menu

# Explicit currency and custom output, if desired
.\.venv\Scripts\python.exe -m app.services.structure_menu --currency MYR --output data/structured/menu.json
```

The default input is `data/qwen-ocr/`, and the default output is `structure.json`
in the working directory. The generator respects the latest OCR `summary.json`,
rejects a batch with failures, and loads only the documents listed as processed.
If a folder has no summary, it reads its JSON files as Qwen OCR documents.

## Structure

- Document metadata: model, model digest, prompt version, source count, item
  occurrence count and explicitly configured currency (otherwise `null`).
- `sources`: image and OCR filenames/hashes, the original transcription, and
  existing OCR uncertainty notes. Python attaches these without model rewriting.
- Each source's `menu.items`: name, printed category/section, description,
  variants and notes about missing or ambiguous information.
- Each variant: size, temperature (`hot`, `iced` or `null`), price and optional
  price condition. Prices are JSON numbers. Missing prices remain `null`.
- `menu.addons`: extra shot, oat milk and other surcharges kept separate from
  menu-item base prices, with an `applies_to` list when the text makes scope clear.

Multiple variants represent multiple size or temperature prices. A description-only
product may have an empty variants list, or a variant with `price: null` when other
variant details are explicit. A variable price such
as filter coffee depending on origin uses `price_note` instead of an invented
number. House blend and seasonal sections remain distinguishable.

Sources remain separate deliberately. A description poster and a price list may
both mention the same drink; their occurrences are not yet a deduplicated menu
database. Joining them by name alone can incorrectly combine different blends,
sizes or recipes. The [catalog-building step](menu-catalog.md) now combines these
records using explicit review decisions while preserving this raw extraction.

## Validation and failure behavior

Pydantic rejects malformed records and negative/nonfinite prices. Original
transcriptions and hashes are attached in Python so the model does not need to
reproduce source quotations. This preserves provenance, not proof that the model
assigned a price correctly. Empty extraction from nonempty menu text fails
explicitly; empty OCR can produce an empty page with a note.

Empty optional description/category/section/size strings are normalized to `null`.
A single add-on applicability string is normalized to a one-element list, while
an existing list is retained. These are generic formatting conversions, not
image-specific parsers or corrections to prices.

The generator makes at most one corrective retry for invalid schema or empty
extraction from nonempty menu text. Incomplete model responses also
fail. `structure.json` is replaced only
after the entire batch succeeds; on failure an existing file remains unchanged,
and the command exits with code 1. No partial batch is published as complete.

Validated intermediate pages are cached in `data/structure-cache/`. The key
includes the OCR JSON, model digest, prompt and output shape; cached pages are
validated again on load. Use `--refresh` to regenerate them. An interrupted batch
can therefore reuse earlier successful pages without repeating every model call.

OCR verification remains deferred. This step reads only OCR JSON, never rechecks
the original images, and cannot recover layout or digits already lost by OCR.
It also cannot prove that every product was extracted or that no product was
misinterpreted. Test names and prices before using the data in customer responses.
