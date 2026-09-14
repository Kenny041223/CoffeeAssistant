# Qwen product embeddings

This stage reads the existing `structure.json` and embeds each product's
`search_text` once using
[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
It preserves all 1,024 dimensions and normalizes each vector for cosine search.
Menu generation continues to use Ollama; this separate embedding stage uses
Hugging Face and PyTorch directly.

**Implementation status:** run successfully on the GTX 1070 Ti PC -- 21
product vectors generated in ~16s on CUDA, independently verified (schema,
menu-hash match, normalized 1024-dim vectors) before being synced to Pinecone.

## First-time installation on the GTX 1070 Ti PC

Use Windows with Python 3.12, a working NVIDIA driver, and your actual generated
`structure.json`. Create a separate environment so the existing OCR environment
keeps its installed dependencies:

```powershell
py -3.12 -m venv generate_embedding/.venv-embeddings
.\generate_embedding\.venv-embeddings\Scripts\python.exe -m pip install torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\generate_embedding\.venv-embeddings\Scripts\python.exe -m pip install -r generate_embedding\requirements-embeddings.txt
```

The setup pins PyTorch 2.8.0 with CUDA 12.6, Sentence Transformers 5.1.2, and
Transformers 4.57.6. The official PyTorch index publishes the Windows Python 3.12
CUDA 12.6 wheel, and the selected library versions satisfy Qwen's documented
requirements. These are source-checked compatibility choices, not a tested
installation on your hardware. See the [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v280),
[wheel index](https://download.pytorch.org/whl/cu126/torch/), and
[Sentence Transformers dependency declarations](https://github.com/huggingface/sentence-transformers/blob/v5.1.2/pyproject.toml).

Keep the CUDA 12.6 pin for this Pascal GPU. CUDA 13 builds drop support for
Pascal, so installing an unrestricted newer PyTorch build can break GPU
execution. [PyTorch compatibility notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)

## One-time model cache preparation

After installation, cache the pinned Hugging Face model on the GTX 1070 Ti PC.
If the snapshot is not already cached, this optional direct Python invocation
downloads it and generates the first vectors from your reviewed menu:

```powershell
.\generate_embedding\.venv-embeddings\Scripts\python.exe -m generate_embedding.embed_menu --input structure.json --output generate_embedding/data/menu-embeddings.json
```

The model is fixed to commit
[`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/commit/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3)
so document vectors and later query vectors use the same weights. A missing
snapshot with `--local-files-only` stops the run rather than downloading it.

## Normal runs with the cached model

Run from the project directory on that PC:

```powershell
.\generate_embedding\generate_embeddings.ps1

# Custom menu and a smaller batch
.\generate_embedding\generate_embeddings.ps1 -InputFile structure.json -Output generate_embedding/data/menu-embeddings.json -BatchSize 1
```

The script uses `generate_embedding/.venv-embeddings/Scripts/python.exe` and always passes
`--local-files-only` to Python. It does not install packages or download models.
Missing dependencies or an incomplete model cache cause an error; complete the
separate first-time setup before retrying. The default input is `structure.json`
and output is `generate_embedding/data/menu-embeddings.json`.

| Setting | Default | Purpose |
|---|---|---|
| `-BatchSize` | `2` | Keep the initial memory demand modest |
| `-MaxLength` | `512` | Maximum tokens per product or query, including its prompt |
| `-Device` | `cuda` | Require an available compatible GPU |
| Precision | FP32 | Avoid relying on BF16 or mixed precision on Pascal |
| Attention | Eager | Use standard attention without FlashAttention |
| Padding | Left | Follow the Qwen embedding input layout |
| Dimensions | `1024` | Keep the native vector size |

CUDA availability and compatibility are checked before model download; failure
does not silently select the CPU. Explicit `-Device cpu` is available for a
deliberate CPU deployment on another machine. No execution is needed on this
laptop now.

An input longer than `-MaxLength` is rejected instead of silently losing menu
details. Review that product's `search_text`, or raise the limit deliberately
(up to 32,768) when the GPU has sufficient memory. If GPU memory runs out, first
close other model workloads and retry with `-BatchSize 1`. Actual speed and
memory use still need measurement on the target PC.

## Output and later retrieval

`structure.json` stays the only menu dataset. `generate_embedding/data/menu-embeddings.json` is a
derived vector artifact containing product IDs, vectors, text hashes, the input
menu hash, and embedding configuration. It references the menu instead of
duplicating product descriptions, price variants, or OCR evidence. Only products
are embedded at this stage; series and add-ons remain in the canonical menu.

The stage validates the menu and vector output and publishes the vector file
atomically after success. A failed run leaves any previous vector file intact.
Regenerate the artifact after changing the menu and use its hashes to prevent
mixing vectors with a different menu version. Future database imports must also
remove products that disappeared from the latest menu.

`QwenEmbedder` in `generate_embedding/embed_menu.py` exposes `encode_documents()` and
`encode_query()` for a later retrieval service or Langflow component. Document
text is encoded without a query instruction. `encode_query()` applies the
shared menu-retrieval instruction using Qwen's `Instruct: ...\nQuery: ...`
format. Both paths use the same model snapshot, normalization, and dimensions.
[Qwen usage guidance](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#usage)

A vector database now stores these vectors under `product_id` for retrieval --
see [pinecone.md](pinecone.md). Apply price, size, and temperature constraints
to structured variants; semantic similarity alone does not enforce them. A
search endpoint, retrieval-augmented querying, and the customer-facing
chatbot remain future work.

Before using recommendations, run the prepared tests on the other PC and assess
top results for real queries, including similar drinks, multilingual preferences,
and price constraints. Neither model reputation nor valid vector dimensions
establish retrieval quality for this menu.
