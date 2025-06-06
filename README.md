# 🧠 GenAI Agent Core (RAG + Local Mistral-7B)

This project demonstrates a fully local Retrieval-Augmented Generation (RAG) pipeline using a Hugging Face transformer (Mistral-7B-Instruct) and LangChain with FAISS. It allows you to embed PDF documents and ask natural language questions grounded in that data.

---

## ✅ Features

- Local LLM inference using `mistralai/Mistral-7B-Instruct-v0.1`
- PDF document embedding with `sentence-transformers`
- Vector search via FAISS
- GPU support (CUDA or fallback to CPU)
- Hugging Face token-based model access
- CLI interface for asking questions about your documents

---

## 🚀 How to Use

### 1. Install Dependencies

Activate your Python environment and install from `requirements.txt`.

```bash
pip install -r requirements.txt
```

Make sure `sentencepiece` is installed:
```bash
pip install sentencepiece
```

### 2. Set Your Hugging Face Token

Ensure your Hugging Face token is set (for gated models like Mistral):

```bash
export HUGGINGFACE_HUB_TOKEN=your_hf_token_here
```

You can add this to your `~/.zshrc` or `~/.bashrc` for persistence.

---

### 3. Embed Your Documents

Put PDFs in the `data/` folder and run:

```bash
python app/embedder.py
```

This creates a FAISS vector index in the `vectorstore/` directory.

---

### 4. Ask a Question (Locally)

```bash
python app/query.py "What is covered in the first section of the PDF?"
```

The script:
- Retrieves top-k document chunks from FAISS
- Constructs a prompt
- Sends the prompt to a local Mistral model
- Prints the generated answer

---

## 💡 Model Caching

The model is stored in Hugging Face's global cache:
```bash
~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1
```

To make it portable or offline-first, you can use `cache_dir="models/mistral-7b"` and add `models/` to your `.gitignore`.

---

## 🛡️ Security & Offline Use

Once the model and tokenizer are downloaded:
- You can run completely offline
- Set `local_files_only=True` in `from_pretrained()` to enforce this

---

## 📁 Project Structure

```
genai-agent-core/
├── app/
│   ├── embedder.py       # Converts PDFs into vector embeddings
│   └── query.py          # Loads model + FAISS, runs inference
├── data/                 # Place PDFs here
├── vectorstore/          # Generated FAISS index
├── models/               # Optional manual model storage
└── requirements.txt
```

---

## 🧠 Future Ideas

- GPT-4 fallback for hallucination checks
- LangChain agents for structured querying
- Streamlit UI or FastAPI backend
