# Qwen menu image extraction

The active model is `qwen3-vl:4b-instruct`, the quantized Ollama distribution of
Qwen3-VL-4B-Instruct. It reads images directly; RapidOCR is no longer used by the
extractor. We use Ollama for local inference on this Windows machine instead of
loading the unquantized Hugging Face weights with Transformers.

References: [model](https://ollama.com/library/qwen3-vl:4b-instruct),
[Windows runtime](https://docs.ollama.com/windows),
[structured vision output](https://docs.ollama.com/capabilities/structured-outputs).

## Setup

The current workspace has a Python virtual environment and a portable Ollama
runtime under `.tools/ollama/`. On another machine, install Python 3.12 and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the official `ollama-windows-amd64.zip` from the
[Ollama v0.33.3 release](https://github.com/ollama/ollama/releases/tag/v0.33.3)
and extract it so `.tools/ollama/ollama.exe` exists. Include the bundled GPU
libraries. Alternatively, use an existing Ollama installation and pass its local
address through `--ollama-url`; ensure the specified model is pulled there first.

```powershell
.\scripts\start-qwen.ps1 -PullModel
.\.venv\Scripts\python.exe -m app.services.ocr --input image
```

The helper starts a hidden local server at `127.0.0.1:11435`, stores model weights
in `.tools/ollama-models/`, and limits parallelism to one image. Model downloads
need internet access and several GB of disk space. Once cached, image inference
is local and does not need an API key. The helper leaves the server running;
the model is set to unload after ten idle minutes. Server logs live in `.tools/`.

## Commands

```powershell
# One image
.\.venv\Scripts\python.exe -m app.services.ocr --input "image/WhatsApp Image 2026-09-08 at 10.25.06 PM.jpeg"

# Existing Ollama installation using its usual port
.\.venv\Scripts\python.exe -m app.services.ocr --ollama-url http://127.0.0.1:11434

# Save an experiment separately
.\.venv\Scripts\python.exe -m app.services.ocr --output data/qwen-ocr/experiment-2

# Tests (mock inference; no model download)
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Inputs are a single image or a nonrecursive folder. Supported extensions are
JPEG, PNG, WebP, BMP and TIFF. Only the first page/frame is read. EXIF orientation
is applied before encoding the RGB image for Ollama. Images are not intentionally
resized by this application; the model runtime performs its own preprocessing.

Inference uses temperature 0, an 8,192-token context, and up to 4,096 generated
tokens. Each image has a configurable `--timeout` of 600 seconds by default.
Truncated responses, malformed JSON and server errors fail explicitly. The
script continues other images and exits nonzero if any failed.

## Output and scope

`data/qwen-ocr/` contains one JSON and TXT per image. `summary.json` records the
latest run, including failures. Repeated runs overwrite successful results for
matching filenames; old results are not deleted if a later run fails. Use the
summary to identify current successes, or choose a fresh output directory.

Schema version 2 contains a transcription string, model uncertainty notes,
source image/hash, model tag/digest, duration, and `verified: false`. The prompt
requests Markdown tables to retain row/column structure, but the model may still
return plain text rows. No box coordinates are invented.
The model can still misplace prices, omit words, or hallucinate. Pydantic checks
the response shape, not whether it matches the image. Empty uncertainty notes
are not a correctness guarantee.

## Initial runtime check

The first complete Qwen batch processed all 14 supplied images with zero request
failures, taking 84.4 seconds with the model already loaded. All 14 unit tests
passed. These are runtime/schema checks, not OCR accuracy measurements.

On this machine, Ollama 0.33.3 detected the GTX 1070 Ti (8 GB) through its CUDA 12
backend. The first full menu request completed in approximately 47 seconds,
including startup. This establishes that inference runs, not that extraction is
accurate. A spot-check found incorrect price-row associations: the first output
attached matcha's iced `20.0` to 5oz rather than 8oz and misaligned mocha prices.
It also returned plain rows despite the request for Markdown tables. Keep these
limitations visible when designing the later extraction/verification stage.

The previous `app/models/ocr.py`, verifier and tests remain as legacy material.
The old `data/ocr/` results have been deleted. The legacy verifier is not used to verify Qwen output. Verification is
deliberately deferred; the new extractor never auto-approves its output.
