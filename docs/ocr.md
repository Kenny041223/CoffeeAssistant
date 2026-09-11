# Qwen setup and menu image transcription

The pipeline uses `qwen3-vl:4b-instruct` to transcribe images, followed by
`qwen3:4b-instruct-2507-q4_K_M` to generate the consolidated menu from all OCR
text. Both run locally through Ollama. Download models and run the pipeline on
the model-capable PC. No scripts or models were executed for this update.

## First-time Python setup

Install Python 3.12, then run from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a new virtual environment on the other PC instead of copying `.venv`.
Copy the source project and `image/`; also copy `data/qwen-ocr/` if you want to
reuse earlier OCR. OCR results and model weights are ignored by Git and will
not be present in a fresh clone.

## First-time Ollama setup and model downloads

Choose either a normal Ollama installation or the project's portable helper.
See the official [Ollama Windows guide](https://docs.ollama.com/windows) for
installation and the standalone archive.

For a normal installation, start Ollama and download both model tags:

```powershell
ollama pull qwen3-vl:4b-instruct
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

For the portable setup, extract the official Windows archive so
`.tools/ollama/ollama.exe` and its bundled GPU libraries exist. Then run:

```powershell
.\scripts\start-qwen.ps1 -PullModel -PullTextModel
```

The helper starts a hidden local server at `127.0.0.1:11435`, stores weights
in `.tools/ollama-models/`, and limits parallel requests and loaded models to one.
Downloads require internet access and disk space. Once the weights are present,
the local pipeline does not require an API key. Server logs live in `.tools/`;
the helper leaves the server running.

## Generate the menu with installed models

The run script uses the existing Python environment and models. It does not
install Ollama or packages, download models, or start the server. Start the
normal Ollama installation first, or start the portable server with
`.\scripts\start-qwen.ps1` (without pull flags).

```powershell
# Normal Ollama installation: port 11434 is the script default
.\scripts\generate_structureFile.ps1

# Portable server: specify its separate port
.\scripts\generate_structureFile.ps1 -OllamaUrl http://127.0.0.1:11435

# Reuse existing OCR results
.\scripts\generate_structureFile.ps1 -SkipOcr
```

A missing model produces an error; the script does not pull it automatically.

## Run OCR independently

```powershell
# All images in the image folder
.\.venv\Scripts\python.exe -m app.services.ocr --input image --ollama-url http://127.0.0.1:11434

# One image
.\.venv\Scripts\python.exe -m app.services.ocr --input "image/WhatsApp Image 2026-09-08 at 10.25.06 PM.jpeg" --ollama-url http://127.0.0.1:11434

# Portable Ollama server
.\.venv\Scripts\python.exe -m app.services.ocr --ollama-url http://127.0.0.1:11435

# A separate output folder for an experiment
.\.venv\Scripts\python.exe -m app.services.ocr --output data/qwen-ocr/experiment-2 --ollama-url http://127.0.0.1:11434

# Automated tests use mock inference; no model is loaded
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Inputs are a single image or a nonrecursive folder. Supported extensions are
JPEG, PNG, WebP, BMP, and TIFF. Only the first page or frame is read. EXIF
orientation is applied before encoding an RGB image. This application does not
intentionally resize images; the runtime performs its own preprocessing.

OCR defaults to temperature 0, `--num-ctx 8192`, `--num-predict 4096`, and
`--timeout 600` seconds per image. The wrapper exposes these as `-OcrNumCtx`,
`-OcrNumPredict`, and `-OcrTimeout`. The separate menu-generation step has a
larger context and output budget; see [menu generation](menu-structure.md).

Truncated responses, malformed JSON, and server errors fail explicitly. OCR
continues processing the remaining images and exits nonzero if any failed. The
wrapper stops before menu generation when OCR fails.

## Output and limitations

`data/qwen-ocr/` contains one JSON file per image. `summary.json` records the
latest run and any failures. Repeated runs overwrite successful results for
matching filenames; old results can remain when a later attempt fails. The
menu generator uses the summary to determine the current batch and rejects
failed batches. Use a fresh output directory for a separate experiment.

OCR schema version 2 contains transcription, model uncertainty notes, source
image and hash, model tag and digest, duration, and `verified: false`. The prompt
requests Markdown tables to retain layout. It does not invent measured bounding
boxes or calibrated confidence scores.

OCR can omit words, misread digits, or misalign price columns. Schema validation
does not check agreement with the image, and empty uncertainty notes do not
establish correctness. The menu-generation model reads this OCR evidence; it
cannot reliably recover information already lost during transcription. Inspect
the images and generated menu before evaluating customer-facing answers.
