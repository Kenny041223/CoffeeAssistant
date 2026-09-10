# Checking OCR correctness

> Deferred: the active extractor now uses Qwen and schema version 2. This older
> verifier supports only RapidOCR schema version 1 in `data/ocr/`. Do not use it
> to approve the new `data/qwen-ocr/` output. Verification will be redesigned later.

No OCR or LLM can guarantee that an image was transcribed perfectly. Valid JSON,
high confidence, and agreement between two models are useful checks, but they do
not establish truth. The most reliable reference for these 14 images is a human
transcription checked against the source, with special attention to prices.

## Pipeline

```text
Image + raw OCR JSON
  -> Schema, image hash, dimensions and region checks
  -> Optional vision model compares image and OCR (second opinion)
  -> Local HTML report with source image, boxes and text
  -> Human transcription + full-image scan for missed text
  -> Exact comparison with the human reference
  -> matches_human_reference OR needs_human_review OR invalid
```

The verifier does not alter OCR output, invent corrections, or insert anything
into the database. `matches_human_reference` means equality to the supplied
reference after whitespace normalization, not a mathematical accuracy guarantee.
The reference is a local reviewer attestation, not an authenticated approval system.

## Run locally

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m app.services.verify_ocr
```

Outputs:

- `data/verification/*.html`: self-contained visual reports; open in a browser.
- `data/verification/*.json`: checks and optional AI findings.
- `data/verification/summary.json`: latest batch statuses.
- `data/references/*.json`: blank human-reference templates, created only if absent.

Exit code 1 means at least one document remains unverified or invalid. This is
expected on the first run. Exit code 0 requires every document to match a complete
human reference and have no blocking checks or AI disagreements. Use distinct
`--input`, `--output`, and `--references` directories. Reports are replaced on rerun;
references are never automatically overwritten.

## Human review

Open a report and its matching reference JSON. For each zero-based region ID,
fill `expected_text` by reading the image, not copying the OCR. Keep punctuation
and decimal digits exact. If OCR detected a logo/noise with no readable text, use
an empty string: that produces a mismatch requiring correction or exclusion in a
later curated dataset. Do not silently bless noise as correct.

Scan the whole image for text that has no OCR box. Record each omitted string and
its location in `missing_text`. Only after that check, set `full_image_reviewed`
to `true` and fill `reviewer` with your name. Rerun the verifier.

A missing or differing transcription keeps the original OCR unverified. Maintain
any corrected OCR as a separate version, preserving raw output. After changing
OCR or the image, create a fresh reference in a new `--references` folder and
review again: hashes deliberately invalidate old approvals.

## Optional LLM vision review

The adapter uses the OpenAI Responses API and a strict JSON output schema.
It sends the source image plus OCR text and coordinates to OpenAI, so it requires
an API key, network access and API usage charges. The default local run sends
nothing. Set `OPENAI_API_KEY` through your local environment/secret manager;
do not paste the key into chat or commit it. Choose an available model supporting
image input and structured output, then run one image first:

```powershell
.\.venv\Scripts\python.exe -m app.services.verify_ocr --input "data/ocr/WhatsApp Image 2026-09-08 at 10.25.06 PM.jpeg.json" --vision-model YOUR_MODEL_ID
```

The model marks each OCR region as match, mismatch, uncertain, or not_text and
reports potentially missed text. It sees the original image, not just JSON.
Every region ID must be covered exactly once. Failures, refusals and incomplete
responses cannot count as a successful review. Findings and model metadata are
saved; suggestions never overwrite raw OCR. Repeated AI runs make new API calls.

AI review is advisory and can share the OCR model's mistakes or be biased by its
output. For a stronger benchmark, have a person transcribe images independently
before looking at OCR. Even a strict output schema only constrains structure.
See the official [vision limitations](https://developers.openai.com/api/docs/guides/images-vision)
and [structured output guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## Before using structured menu JSON

This verifier checks the current OCR-region JSON. It does not yet certify the
future menu-item JSON. That stage also needs to check:

- Every item and variant against the image, including omissions and duplicates.
- Each price attached to the correct name, size and hot/iced column. Merely finding
  `18.0` somewhere in an image is insufficient.
- Add-on prices separated from base prices; missing values kept unknown.
- Names and descriptions consistent across posters and tables, without assuming
  two similar names are the same item.

Build a reviewed benchmark, then measure character/word error rates, missing-item
recall, exact price accuracy and variant-association accuracy on held-out images.
This initial verifier reports discrepancies, not aggregate OCR accuracy metrics.
