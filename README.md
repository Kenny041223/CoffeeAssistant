# ☕ Coffee Shop AI Customer Service Agent

An AI-powered customer service system designed to answer coffee shop customers' questions about menu items, prices, recommendations, store information, and other frequently asked questions.

The project combines **backend software engineering, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), structured data extraction, and AI tool calling** to explore how AI can automate real-world customer service workflows.

> **Project Status:** 🚧 In Development

### Current milestone — Qwen OCR and structured menu extraction

The active extractor uses **Qwen3-VL-4B-Instruct**, running locally through
Ollama. It transcribes menu images into text (including Markdown tables) inside
Pydantic-validated JSON. A second text-only Qwen step maps those transcriptions
into menu items, variants and add-ons using one shared schema. No API key is required.

From this project directory:

```powershell
# Start the local runtime; download the model if needed
.\scripts\start-qwen.ps1 -PullModel

# Transcribe the image folder
.\.venv\Scripts\python.exe -m app.services.ocr --input image

# Download the text model once, then generate structure.json
.\scripts\start-qwen.ps1 -PullTextModel
.\.venv\Scripts\python.exe -m app.services.structure_menu
```

Results are saved in `data/qwen-ocr/` as `.json` and `.txt`, with a batch
`summary.json`. Each JSON includes the model digest, source hash, transcription,
uncertainty notes, duration and `verified: false`. Qwen does not provide measured
bounding boxes or calibrated OCR confidence scores; those fields are not fabricated.

See [Qwen setup and usage](docs/ocr.md) for fresh installation and single-image commands.
RapidOCR and ONNX Runtime are no longer required by the active extractor.

`structure.json` is an LLM-generated draft grouped by source image. Each entry has
name, category, section, description and size/temperature/price variants. Original
transcriptions are attached by Python. Missing values are `null`; add-ons are separate. The default currency is
unspecified; use `--currency MYR` when that currency has been confirmed. See
[the structuring workflow](docs/menu-structure.md) for schema and behavior.

**Verification is deferred.** The previous verifier remains available for old
RapidOCR schema version 1 only. It is not connected to Qwen schema version 2.

---

## 📌 Problem

Coffee shop staff frequently answer repetitive customer questions such as:

- What drinks are available?
- Which drinks are below a certain price?
- Do you have non-coffee drinks?
- What would you recommend if I don't like sweet drinks?
- What are the store opening hours?
- Are there any current promotions?

Answering these questions manually takes staff time, especially during busy periods.

This project aims to build an AI customer service system capable of answering these questions using actual store information while reducing hallucinations and maintaining reliable responses.

---

## 🎯 Project Goals

The goal is not simply to build a chatbot.

The project aims to explore how an AI system can be integrated into a real software architecture while considering:

- response accuracy
- hallucination prevention
- structured data extraction
- retrieval quality
- tool usage
- API design
- latency
- LLM usage cost
- error handling
- evaluation
- deployment

---

## 🏗️ Planned Architecture

```text
Customer
   │
   ▼
Messaging Interface / Web Client
   │
   ▼
FastAPI Backend
   │
   ▼
AI Orchestrator
   │
   ├── Menu Database
   │
   ├── RAG Pipeline
   │      ├── Embeddings
   │      └── Vector Database
   │
   ├── Agent Tools
   │      ├── Search Menu
   │      ├── Store Information
   │      └── Promotions
   │
   └── LLM
          │
          ▼
     Generated Response
```

---

## 🧠 Planned Features

### 1. Menu Image Extraction

The existing coffee shop menu will be converted from an image into structured data using OCR and/or a vision-capable AI model.

```text
Menu Image
    ↓
OCR / Vision Model
    ↓
Structured Extraction
    ↓
Pydantic Validation
    ↓
JSON
    ↓
Database
```

Example output:

```json
{
  "name": "Iced Matcha Latte",
  "category": "non_coffee",
  "price": 13.90,
  "temperature_options": ["iced"],
  "description": "Matcha with milk"
}
```

---

### 2. Menu API

FastAPI will expose REST endpoints for accessing menu information.

Planned endpoints include:

```http
GET /menu
GET /menu/{item_id}
GET /menu?category=coffee
GET /menu?max_price=15
```

---

### 3. AI Customer Assistant

Customers will be able to ask questions using natural language.

Example:

```text
Customer:
I want something cold without coffee under RM15.

Assistant:
Based on the available menu, here are some options...
```

The AI system will determine what information is required and retrieve relevant data before generating its response.

---

### 4. Retrieval-Augmented Generation (RAG)

Unstructured store knowledge may include:

- FAQs
- store information
- product descriptions
- promotions
- policies
- other relevant documents

Documents will be converted into embeddings and stored in a vector database.

```text
Customer Question
       ↓
Embedding
       ↓
Vector Search
       ↓
Relevant Context
       ↓
LLM
       ↓
Grounded Response
```

This allows the model to retrieve relevant information instead of placing the entire knowledge base into every prompt.

---

### 5. AI Tool Calling

The AI assistant will have access to controlled application functions.

Potential tools include:

```python
search_menu()
get_product_details()
get_store_hours()
get_current_promotions()
check_item_availability()
```

The LLM will decide when a tool is required, while the backend remains responsible for executing the actual operation.

---

### 6. Evaluation

The project will include an evaluation dataset containing realistic customer questions.

Examples:

```text
"What drinks are below RM10?"
"Do you have anything without coffee?"
"What time does the store close?"
"Recommend something that isn't too sweet."
"What promotions are available?"
"Do you sell pizza?"
```

Potential evaluation metrics include:

- answer accuracy
- retrieval accuracy
- tool selection accuracy
- hallucination rate
- response latency
- average LLM cost per request

Evaluation results will be added once the system has been implemented and tested.

---

## 🛠️ Planned Technology Stack

### Backend

- Python
- FastAPI
- Pydantic

### AI

- Large Language Model API
- Embeddings
- Retrieval-Augmented Generation (RAG)
- Tool / Function Calling

### Database

- Pinecone
- Vector database

### Infrastructure

- Docker
- Cloud deployment

### Future Integration

- Messaging platform / social media API

---

## 📂 Proposed Project Structure

```text
coffee-ai-agent/
│
├── app/
│   ├── api/
│   │   ├── chat.py
│   │   ├── menu.py
│   │   └── webhook.py
│   │
│   ├── agents/
│   │   ├── customer_agent.py
│   │   └── tools.py
│   │
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── retrieval.py
│   │   └── ingestion.py
│   │
│   ├── models/
│   ├── services/
│   ├── database/
│   └── main.py
│
├── evaluation/
├── tests/
├── docs/
│   └── architecture.md
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The project structure may change as development progresses.

---

## 🚀 Development Roadmap

### Phase 1 — Menu Data Pipeline

- [x] Collect menu image
- [x] Implement and run a local OCR baseline
- [x] Extract menu information (LLM draft)
- [x] Convert extracted information into structured JSON
- [x] Validate data structure using Pydantic
- [ ] Store menu data
- [ ] Build menu REST API

### Phase 2 — AI Assistant

- [ ] Create `/chat` endpoint
- [ ] Integrate LLM API
- [ ] Implement structured responses
- [ ] Connect assistant with menu data

### Phase 3 — RAG

- [ ] Prepare store knowledge base
- [ ] Generate embeddings
- [ ] Configure vector database
- [ ] Implement retrieval pipeline
- [ ] Ground responses using retrieved context

### Phase 4 — Tool Calling

- [ ] Implement agent tools
- [ ] Connect tools to backend services
- [ ] Handle tool calls
- [ ] Add tool error handling

### Phase 5 — Evaluation & Production Engineering

- [ ] Create evaluation dataset
- [ ] Measure response accuracy
- [ ] Measure hallucination rate
- [ ] Measure latency
- [ ] Track LLM token usage and cost
- [ ] Add logging
- [ ] Add error handling and retries
- [ ] Containerize application with Docker

### Phase 6 — Deployment & Integration

- [ ] Deploy backend
- [ ] Connect messaging platform
- [ ] Add webhook integration
- [ ] Perform end-to-end testing

---

## 📊 Results

Evaluation results will be published here after implementation.

| Metric | Result |
|---|---|
| Answer Accuracy | TBD |
| Tool Selection Accuracy | TBD |
| Retrieval Accuracy | TBD |
| Hallucination Rate | TBD |
| Average Response Latency | TBD |
| Average LLM Cost / Request | TBD |

---

## 💡 Engineering Questions

Throughout development, this project will explore questions such as:

- When should structured SQL queries be used instead of vector retrieval?
- Which information should be stored as structured data versus embedded documents?
- How can hallucinations be detected or reduced?
- When should the LLM call a tool instead of answering directly?
- How should the system behave when required information is unavailable?
- How can retrieval quality be evaluated?
- What trade-offs exist between response quality, latency, and cost?
- When should a human review an AI-generated response?

These decisions and findings will be documented as the project evolves.

---

## ⚠️ Current Limitations

This project is currently under active development.

Features and architecture described above represent the planned direction of the system and may change as technical requirements and real-world constraints are discovered.

---

## 📖 What I Hope to Learn

This project is being developed to strengthen practical experience in:

- AI application engineering
- backend API development
- LLM integration
- RAG systems
- AI agents and tool calling
- database design
- AI evaluation
- production deployment
- software architecture

The long-term objective is to understand not only how to call an LLM API, but how to design, evaluate, and deploy a reliable AI-powered software system.
