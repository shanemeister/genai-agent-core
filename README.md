# 🧠 GenAI Agent Core — RAG Pipeline with Mixtral, LLaMA3, and OpenAI

This project implements a production-ready Retrieval-Augmented Generation (RAG) assistant using **LangChain**, **FAISS**, and three model options: **Mixtral (GGUF)**, **LLaMA3 (Hugging Face Transformers)**, and **GPT-4o (OpenAI)**. It supports both CLI and FastAPI interfaces for real-time question answering over embedded documents.

> ✅ Designed for technical interviews, enterprise search, or AI assistant prototyping — fully offline-capable and extensible.

⚠️ Note: This system requires a rather beefy workstation/server. I developed it using a system with the following specs:

* CPU: Threadripper 64 Core, 128 Threads
* Memory: 512 GB DDR5
* Disk: 20 TB NVMe SSD
* GPU: Two Nvidia A6000 GPUs


---

## ✅ Key Features

* 🔍 **RAG architecture**: combines retrieval + generation for grounded, explainable answers.
* 🤖 **Local LLM inference**: runs via `llama.cpp` or Hugging Face Transformers.
* 🧠 **Semantic vector search**: uses `sentence-transformers` and `FAISS`.
* 📆 **FastAPI backend**: exposes `/ask` and `/rebuild` endpoints with JSON responses.
* 💪 **Session memory**: persists chat history in PostgreSQL.
* 📝 **Document ingestion**: supports `.pdf`, `.docx`, and `.txt` files via directory-based loader.
* 🧪 **Swagger docs**: self-documenting API.
* 🔐 **Offline mode**: models can run fully locally with no external API dependencies.
* 🔢 **Token estimation**: optional for local models like LLaMA3 and Mixtral.
* 🧾 **Model logging**: API metadata includes model used and latency.

---

## 📁 Project Structure

```
genai-agent-core/
├── app/
│   ├── interface/           # CLI tool for local RAG querying
│   │   └── query_plus.py    # Full-featured vector+LLM querying
│   └── llm_core.py          # Basic model call wrappers
├── chat/
│   ├── postgres_history.py  # Session-aware chat logging to PostgreSQL
│   └── vectorstore_memory.py# FAISS vectorstore + ingestion
├── configs/
│   └── llm_config.yaml      # Model config (OpenAI fallback)
├── data/                    # Place PDFs, DOCX, and TXT files here
├── vectorstore/             # Saved FAISS index
├── models/                  # Local GGUF model storage (Mixtral) and HF model (LLaMA3)
├── rag_api_service.py       # FastAPI app with /ask and /rebuild
├── search_vectorstore.py    # CLI tool for searching FAISS chunks
├── rebuild_vs.py            # Manual vectorstore rebuild runner
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
conda activate genai-core
pip install -r requirements.txt
pip install sentencepiece
```

---

### 2. Add your documents

Drop your `.pdf`, `.docx`, or `.txt` files into `data/` (or subfolders).

Then run:

```bash
python rebuild_vs.py
```

This rebuilds the FAISS vector index with semantic embeddings using `sentence-transformers`.

---

### 3. Query via CLI

```bash
python app/interface/query_plus.py "What is RAG?" --model mixtral --chat --session-id demo1
```

Options:

* `--model mixtral|llama3|gpt4o`
* `--filter-tag` or `--filter-file` for scoped retrieval
* `--chat` enables persistent memory
* `--session-id` groups chats by user

---

### 4. Query via FastAPI

Start the API:

```bash
uvicorn rag_api_service:app --reload
```

Visit Swagger docs:

```
http://127.0.0.1:8000/docs
```

Example POST to `/ask`:

```json
{
  "query": "What is retrieval-augmented generation?",
  "session_id": "test1",
  "model": "llama3",
  "token_estimate": true
}
```

Response includes:

* `answer`
* `sources[]`: source text chunks
* `meta`:

  * `elapsed_seconds`
  * `tokens_estimated`
  * `model`

---

## 💡 Prompt Engineering

Prompts are dynamically constructed using:

* Retrieved chunks (`similarity_search_with_score`)
* Optional session memory
* Custom headers: *"You are an AI document analyst..."*

You can customize this in `generate_prompt()` inside `query_plus.py`.

---

## 🧠 Model Options

* **Mixtral**: GGUF model run via `llama.cpp` (fast, parallelized on GPU)
* **LLaMA 3 8B**: Hugging Face Transformers model running in FP16 on GPU
* **GPT-4o**: OpenAI fallback using API + YAML config

To use OpenAI:

```bash
export OPENAI_API_KEY=sk-...
```

---

## 💃 PostgreSQL Chat History

Chat logs are stored in `chathist.chat_history`:

* `session_id`, `role`, `content`, `created_at`
* Used for session memory and context carryover

Uses `.pgpass` for secure auth (no credentials in code).

---

## 📊 Metadata & Monitoring

API responses include:

* `elapsed_seconds`: total inference + retrieval time
* `tokens`: OpenAI usage count if applicable
* `tokens_estimated`: estimated using tokenizer for local models
* `model`: which LLM handled the request

---

## 🧪 Performance Comparison

| Model     | Latency (s) | Local | Token Usage   | Notes                            |
| --------- | ----------- | ----- | ------------- | -------------------------------- |
| Mixtral   | \~2.7       | ✅ Yes | estimated     | Fast + multi-GPU (llama.cpp)     |
| LLaMA3-8B | \~11.2      | ✅ Yes | estimated     | HuggingFace, accurate, slower    |
| GPT-4o    | \~1.5–2.5   | ❌ API | exact via API | Fastest, but external dependency |

---

## 🔢 Testing / Debugging Tools

| Tool               | Command                                      |
| ------------------ | -------------------------------------------- |
| Search vectorstore | `python search_vectorstore.py "your query"`  |
| Rebuild index      | `python rebuild_vs.py`                       |
| API query          | `curl -X POST http://127.0.0.1:8000/ask ...` |

---

## 🔐 Offline-First Capabilities

Set `local_files_only=True` in `AutoTokenizer` and `AutoModelForCausalLM` to enforce offline mode.

---

## 🌐 Roadmap

* ✅ Swagger-enabled API
* ✅ Chat memory with PostgreSQL
* ✅ LLaMA 3 support via Transformers
* ✅ Token estimation for local models
* ✅ Model logging in metadata
* ⏳ User auth or API key guardrails
* ⏳ Vectorstore auto-refresh with S3 uploads
* ⏳ Web frontend (Streamlit or React)
* ⏳ GCP Cloud Run deployment config

---

## 👤 Author

**Randall Shane** — [LinkedIn](https://www.linkedin.com/in/randall-shane/)

Senior AI/ML Systems Architect with 20+ years of experience designing scalable, production-grade platforms across healthcare, finance, and enterprise analytics. Most recently led end-to-end GenAI and retrieval-augmented generation (RAG) development using LangChain, FAISS, and OpenAI/GCP models.

This repository reflects my current focus: building secure, modular, and extensible GenAI pipelines that combine local and cloud-hosted LLMs with vector search, prompt engineering, and human-in-the-loop evaluation. I lead senior-level roles in AI/ML engineering or GenAI systems design, particularly where intelligent assistants and clinical/niche use cases are a focus.