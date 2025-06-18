import os
import sys
import torch
import yaml
import json
import time
from collections import defaultdict
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
#pip install -U langchain-community
from llama_cpp import Llama
from chat.postgres_history import load_chat_history, save_message
from chat.vectorstore_memory import retrieve_similar_context, add_to_vectorstore  # optional
from app.llm_core import basic_query as call_model 
import __main__ as main_mod  
from chat.postgres_history import load_chat_history, save_message

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
conversation_history = []

# Configuration and constants
VECTORSTORE_PATH = "vectorstore"
MISTRAL_GGUF_PATH = "/home/exx/myCode/genai-agent-core/models/mixtral-gguf/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
LLAMA3_MODEL_PATH = "/home/exx/myCode/genai-agent-core/models/llama3-8b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load OpenAI configuration from YAML
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "llm_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
openai_model_name = config["models"]["gpt_4o_mini"]["model"]
llm = ChatOpenAI(model=openai_model_name, temperature=0.3)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# Use a pre-configured tokenizer for GGUF models
llama_tokenizer = Llama(model_path=MISTRAL_GGUF_PATH, n_ctx=4096, n_gpu_layers=0, seed=42)

def safe_token_len(llama_model, text):
    try:
        return len(llama_model.tokenize(text.encode("utf-8")))
    except Exception as e:
        print(f"⚠️ Tokenization failed: {e}")
        return 0
    
def generate_prompt(question, docs):
    context_chunks = []
    total_tokens = 0
    max_tokens = 3500

    for doc in docs:
        content = f"[{doc.metadata.get('source')}] {doc.page_content}"
        num_tokens = safe_token_len(llama_tokenizer, content)
        if total_tokens + num_tokens > max_tokens:
            break
        context_chunks.append(content)
        total_tokens += num_tokens

    context = "\n\n".join(context_chunks)
    return (
        "You are an AI document analyst. Use the context below, which may include multiple files, to answer comprehensively.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

def ask_mistral_gguf(prompt, stream=False, history=None):
    try:
        print("\n💡 Using Mistral GGUF via llama.cpp")
        model = Llama(model_path=MISTRAL_GGUF_PATH, n_ctx=4096, n_gpu_layers=100, seed=42)

        if stream:
            print("📤 Streaming response:")
            response_stream = model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                stream=True
            )
            full_response = ""
            for chunk in response_stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)
                full_response += delta
            print()
            return full_response.strip()
        else:
            if history is None:
                history = []
            history.append({"role": "user", "content": prompt})
            output = model.create_chat_completion(
                messages=history,
                max_tokens=256
            )
            response = output["choices"][0]["message"]["content"].strip()
            print(response)
            history.append({"role": "assistant", "content": response})
            return response

    except Exception as e:
        print(f"⚠️ Mistral GGUF failed to load or generate: {e}\nFalling back to LLaMA 3...")
        return ask_llama3_hf(prompt)

def ask_llama3_hf(prompt):
    try:
        print("\n🧠 Using LLaMA 3 8B (Hugging Face FP16)")
        tokenizer = AutoTokenizer.from_pretrained(LLAMA3_MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA3_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        for k in inputs:
            if inputs[k].dtype == torch.float32:
                inputs[k] = inputs[k].half()
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            repetition_penalty=1.1,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        return response
    except Exception as e:
        print(f"⚠️ LLaMA 3 failed to load or generate: {e}\nFalling back to GPT-4o...")
        return ask_openai(prompt, [])

def ask_openai(question, context):
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set. Cannot use OpenAI fallback.")
        return
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    prompt = generate_prompt(question, context)
    print(f"\n💬 Answer (OpenAI {openai_model_name}):\n")
    response = llm.invoke(prompt, return_usage=True)
    print(response.content)
    print("📊 Token usage:", response.usage)

    return response.content, response.usage

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

    # Build filters
    filter_kwargs = {}
    if filter_tag:
        filter_kwargs["metadata"] = {"tag": filter_tag}
    if filter_filename:
        filter_kwargs.setdefault("metadata", {})["source"] = filter_filename

    # Retrieve documents
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

    # Export chunk metadata for debugging
    chunk_metadata = [
        {
            "rank": int(i + 1),
            "score": float(score),
            "source": str(doc.metadata.get("source", "N/A")),
            "tag": str(doc.metadata.get("tag", "N/A")),
            "content": doc.page_content[:300]
        }
        for i, (doc, score) in enumerate(relevant_docs)
    ]
    chunk_db_path = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "chunk_db.json")
    with open(chunk_db_path, "w") as f:
        json.dump(chunk_metadata, f, indent=2)
    print(f"\n📦 Exported chunk metadata to: {chunk_db_path}")

    # Run the model
    answer = ""
    if model_choice == "gpt4o":
        answer, usage = ask_openai(question, docs)
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

    # Save chat and update memory
    if session_id and answer:
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)
        add_to_vectorstore([question, answer])

    elapsed = round(time.time() - start_time, 2)

    # CLI mode or batch eval mode
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query your local or OpenAI LLM with optional filters.")
    parser.add_argument("question", type=str, help="Your question to ask.")
    parser.add_argument("--model", type=str, choices=["mixtral", "llama3", "gpt4o"], default="mixtral",
                        help="Choose the LLM to use: mixtral (GGUF), llama3 (Transformers), or gpt4o (OpenAI).")
    parser.add_argument("--filter-tag", type=str, help="Filter by semantic tag.")
    parser.add_argument("--filter-file", type=str, help="Filter by source PDF filename.")
    parser.add_argument("--filter-all", action="store_true", help="Use all documents (no similarity filtering)")
    parser.add_argument("--stream", action="store_true", help="Stream response (for Mixtral GGUF)")
    parser.add_argument("--chat", action="store_true", help="Enable conversation memory (persistent across turns)")
    parser.add_argument("--session-id", type=str, help="Session ID for persistent chat history")
    args = parser.parse_args()

    # In CLI mode, rules must be passed (None unless set up externally)
    ask_question(
        question=args.question,
        model_choice=args.model,
        filter_tag=args.filter_tag,
        filter_filename=args.filter_file,
        filter_all=args.filter_all,
        stream=args.stream,
        history=conversation_history,
        session_id=args.session_id,
        chat_enabled=args.chat,
        rules=None
    )