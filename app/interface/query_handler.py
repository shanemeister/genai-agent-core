import os
import sys
import time
import json
import yaml
import torch
import argparse
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from chat.postgres_history import load_chat_history, save_message
from chat.vectorstore_memory import add_to_vectorstore
from app.utils.vector_utils import get_vectorstore
from app.utils.prompt_utils import generate_prompt
from app.llm.inference_runners import ask_llama3_hf, ask_mistral_gguf, ask_openai
from app.configs.global_constants import MISTRAL_GGUF_PATH, VECTORSTORE_PATH, LLAMA3_MODEL_PATH, DEVICE, HF_TOKEN, OPENAI_API_KEY
from llama_cpp import Llama
import __main__ as main_mod

sys.path.append(str(Path(__file__).resolve().parents[2]))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
conversation_history = []


# Load OpenAI model config
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "llm_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# llm = get_openai_llm("gpt_4o_mini")

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


llama_tokenizer = Llama(model_path=MISTRAL_GGUF_PATH, n_ctx=4096, n_gpu_layers=0, seed=42)

def safe_token_len(llama_model, text):
    try:
        return len(llama_model.tokenize(text.encode("utf-8")))
    except Exception as e:
        print(f"⚠️ Tokenization failed: {e}")
        return 0

def ask_question(
    question, model_choice="mixtral",
    filter_tag=None, filter_filename=None, filter_all=False,
    stream=False, history=None,
    session_id=None, chat_enabled=False,
    rules=None
):
    start_time = time.time()
    usage = {}

    if chat_enabled and session_id:
        history = load_chat_history(session_id)
    else:
        history = []

    db = load_vectorstore()
    print("\n🔍 Question:", question)

    filter_kwargs = {}
    if filter_tag:
        filter_kwargs["metadata"] = {"tag": filter_tag}
    if filter_filename:
        filter_kwargs.setdefault("metadata", {})["source"] = filter_filename

    if filter_all:
        all_docs = db.similarity_search("", k=1000)
        relevant_docs = [(doc, 0.0) for doc in all_docs]
    else:
        relevant_docs = db.similarity_search_with_score(question, k=3, **filter_kwargs)

    if not relevant_docs:
        print("⚠️ No relevant documents found.")
        return {"answer": "", "meta": {"elapsed_seconds": 0, "tokens": None}}

    docs = [doc for doc, _ in relevant_docs]
    prompt = generate_prompt(question, docs)

    print("\n📚 Top Retrieved Documents with Scores:")
    for i, (doc, score) in enumerate(relevant_docs):
        snippet = doc.page_content[:120].replace("\n", " ")
        source = doc.metadata.get("source", "N/A")
        tag = doc.metadata.get("tag", "N/A")
        print(f"[{i+1}] Score: {float(score):.4f} | File: {source} | Tag: {tag} | {snippet}...")

    chunk_metadata = [
        {
            "rank": i + 1,
            "score": float(score),
            "source": doc.metadata.get("source", "N/A"),
            "tag": doc.metadata.get("tag", "N/A"),
            "content": doc.page_content[:300]
        }
        for i, (doc, score) in enumerate(relevant_docs)
    ]
    chunk_db_path = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "chunk_db.json")
    with open(chunk_db_path, "w") as f:
        json.dump(chunk_metadata, f, indent=2)
    print(f"\n📦 Exported chunk metadata to: {chunk_db_path}")

    if model_choice == "gpt4o":
            answer, usage = ask_openai(question, docs, model_key="gpt_4o")
    elif model_choice == "llama3":
        answer = ask_llama3_hf(prompt)
    elif model_choice == "mixtral":
        if stream:
            history = history.copy()
            answer = ask_mistral_gguf(prompt, stream=stream, history=history)
        else:
            answer = ask_mistral_gguf(prompt, stream=stream)
    else:
        raise ValueError("Invalid model choice: must be one of ['mixtral', 'llama3', 'gpt4o']")

    if session_id and answer:
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)
        add_to_vectorstore([question, answer])

    elapsed = round(time.time() - start_time, 2)

    if hasattr(main_mod, '__file__') and main_mod.__file__.endswith("query_eval.py"):
        return answer
    else:
        print("\n📊 Evaluation Summary:")
        for rule in rules or []:
            rule_type = rule.get("type")
            rule_val = rule.get("value")
            passed = True
            if rule_type == "contains":
                passed = str(rule_val).lower() in answer.lower()
            elif rule_type == "length_gt":
                passed = len(answer.strip()) > int(rule_val)
            elif rule_type == "regex":
                import re
                passed = re.search(rule_val, answer) is not None
            if passed:
                print(f"✔️ Passed: Rule = {rule}")
            else:
                print(f"❌ Failed: Rule = {rule}")

        return {
            "answer": answer,
            "meta": {
                "elapsed_seconds": elapsed,
                "tokens": usage.get("total_tokens", None)
            }
        }
