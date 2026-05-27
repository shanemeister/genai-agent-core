# 🧠 GenAI Agent Core — RAG Pipeline with Mixtral, LLaMA3, and OpenAI

![CI](https://github.com/shanemeister/genai-agent-core/actions/workflows/ci.yml/badge.svg)

CI: see [.github/workflows/ci.yml](.github/workflows/ci.yml). Currently a stub workflow (real pytest job blocked on NOES-54). Branch protection on `main` requires CI pass before merge — admin (rsync deploy) can still push directly.


This project implements a production-ready Retrieval-Augmented Generation (RAG) assistant using **LangChain**, **FAISS**, and three model options: **Mixtral (GGUF)**, **LLaMA3 (Hugging Face Transformers)**, and **GPT-4o (OpenAI)**. It supports both CLI and FastAPI interfaces for real-time question answering over embedded documents.

⚠️ Note: This is a continuous work in progress as time allows. Please check back frequently for update. 

> ✅ Designed for technical interviews, enterprise search, or AI assistant prototyping — fully offline-capable and extensible.

🚉 Platform: This system is built on a local machine to avoid cloud compute costs. However, it can be easily transported to any cloud such as AWS, GCP, or Azure. To run locally requires a rather beefy workstation/server. I developed it using a system with the following specs:

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
├── app
│   ├── agents
│   │   ├── doc_agent.py
│   │   └── orchestrator.py
│   ├── chains
│   │   └── qa_chain.py
│   ├── configs
│   │   ├── llm_config.yaml
│   │   ├── pdf_tags.yml
│   │   └── prompts
│   ├── embeddings
│   │   └── embedder.py
│   ├── interface
│   │   ├── __pycache__
│   │   │   ├── query_eval.cpython-310.pyc
│   │   │   └── query_plus.cpython-310.pyc
│   │   ├── query_eval.py
│   │   ├── query_plus.py
│   │   └── query.py
│   ├── llm_core.py
│   ├── loaders
│   │   └── fhir_loaders.py
│   ├── retrievers
│   │   └── vector_retriever.py
│   ├── tagging
│   │   └── semantic_tagger.py
│   ├── tools
│   │   ├── __pycache__
│   │   │   └── summarize.cpython-310.pyc
│   │   └── summarize.py
│   └── vectorstore
│       ├── chunk_db.json
│       ├── index.faiss
│       └── index.pkl
├── chat
│   ├── postgres_history.py
│   ├── __pycache__
│   │   ├── postgres_history.cpython-310.pyc
│   │   └── vectorstore_memory.cpython-310.pyc
│   └── vectorstore_memory.py
├── config.py
├── data
│   ├── csv
│   ├── docs
│   │   ├── rag_explained.txt
│   ├── json
│   ├── kindle
│   ├── pdfs
│   │   ├── 🎬
│   └── text
├── eval
│   ├── eval_queries.json 
│   └── eval_queries.jsonl
├── LICENSE
├── main.py
├── models
│   ├── download_mixtral.py
│   ├── llama3-8b
│   │   ├──  🎬
│   ├── mistral-7b
│   │   └── 🎬
│   ├── mixtral-8x7b
│   │   ├── 🎬
│   ├── mixtral-gguf
│   │   ├── 🎬
│   └── model_test.py
├── rag_api
├── rag_api_service.py
├── README.md
├── rebuild_vs.py
├── requirements.txt
├── search_vectorstore.py
├── test.py
└── vectorstore
    ├── index.faiss
    └── index.pkl
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