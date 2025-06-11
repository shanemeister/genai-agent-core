# 🧠 GenAI Agent Core — RAG Pipeline with Local LLM (Mixtral/Mistral)

This project implements a production-ready Retrieval-Augmented Generation (RAG) assistant using **LangChain**, **FAISS**, and a **local LLM (Mistral 7B or Mixtral)**. It supports both CLI and FastAPI interfaces for real-time question answering over embedded documents.

> ✅ Designed for technical interviews, enterprise search, or AI assistant prototyping — fully offline-capable and extensible.


✅ Key Features
	•	🔍 RAG architecture: combines retrieval + generation for grounded, explainable answers.
	•	🤖 Local LLM inference: runs on mistralai/Mistral-7B-Instruct (GGUF via llama.cpp) or HuggingFace models.
	•	🧠 Semantic vector search: uses sentence-transformers and FAISS.
	•	📦 FastAPI backend: exposes /ask and /rebuild endpoints with JSON responses.
	•	🧰 Session memory: persists chat history in PostgreSQL.
	•	📁 Document ingestion: supports .pdf, .docx, and .txt files via directory-based loader.
	•	🧪 Swagger docs: self-documenting API.
	•	🔐 Offline mode: models can run fully locally with no external API dependencies.

⸻

📁 Project Structure

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
├── models/                  # Local GGUF model storage (Mistral/Mixtral)
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
  "token_estimate": true,
  "model": "llama3"
}
```

Response includes:
	•	answer
	•	sources[]: source text chunks
	•	meta: elapsed_seconds, token count (for OpenAI)

⸻

💡 Prompt Engineering

Prompts are dynamically constructed using:

* Retrieved chunks (`similarity_search_with_score`)
* Optional session memory
* Custom headers: *"You are an AI document analyst..."*

You can customize this in `generate_prompt()` inside `query_plus.py`.

⸻

🧠 Model Options
	•	Mixtral/Mistral: supports GGUF via llama.cpp + GPU
	•	LLaMA3 8B: runs via HuggingFace transformers in FP16
	•	GPT-4o: OpenAI fallback with YAML config + API key

To use OpenAI, set:

export OPENAI_API_KEY=sk-...


⸻

🗃️ PostgreSQL Chat History

Chat logs are stored in chathist.chat_history:
	•	session_id, role, content, created_at
	•	Used for session memory and context carryover

Uses .pgpass for secure auth (no credentials in code).

⸻

📊 Metadata & Monitoring

API responses include:
	•	elapsed_seconds: total inference + retrieval time
	•	tokens: OpenAI usage count if applicable

Future additions:
	•	Token estimation for local models
	•	Latency breakdowns
	•	Prometheus metrics

⸻

🧪 Testing / Debugging Tools

Tool	Command
Search vectorstore	python search_vectorstore.py "your query"
Rebuild index	python rebuild_vs.py
API call	curl -X POST http://127.0.0.1:8000/ask ...


⸻

🔐 Offline-First Capabilities

Set local_files_only=True in AutoTokenizer and AutoModelForCausalLM to enforce offline mode.

⸻

🌐 Future Roadmap
	•	✅ Swagger-enabled API
	•	✅ Chat memory with PostgreSQL
	•	⏳ User auth or API key guardrails
	•	⏳ Vectorstore auto-refresh with S3 uploads
	•	⏳ Web frontend (Streamlit or React)
	•	⏳ GCP Cloud Run deployment config

⸻