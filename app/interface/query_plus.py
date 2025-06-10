import os
import sys
import torch
import yaml
import json
from collections import defaultdict
from pathlib import Path
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from llama_cpp import Llama

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

def generate_prompt(question, docs):
    context = "\n\n".join([f"[{doc.metadata.get('source')}] {doc.page_content}" for doc in docs])
    return (
        f"You are an AI document analyst. Use the context below, which may include multiple files, to answer comprehensively.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

def ask_mistral_gguf(prompt, stream=False, history=None):
    try:
        print("\n💡 Using Mistral GGUF via llama.cpp")
        model = Llama(model_path=MISTRAL_GGUF_PATH, n_ctx=4096, n_gpu_layers=100, seed=42)
        output = model(prompt, max_tokens=256)
        print(output["choices"][0]["text"].strip())
        
        if stream:
            print("📤 Streaming response:")
            response_stream = model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                stream=True
            )
            for chunk in response_stream:
                print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
            print()
        else:
            if history is None:
                history = []
            history.append({"role": "user", "content": prompt})
            output = model.create_chat_completion(
                messages=history,
                max_tokens=256
            )
            print(output["choices"][0]["message"]["content"].strip())  
            response = output["choices"][0]["message"]["content"].strip()
            print(response)
            history.append({"role": "assistant", "content": response})              
    except Exception as e:
        print(f"⚠️ Mistral GGUF failed to load or generate: {e}\nFalling back to LLaMA 3...")
        ask_llama3_hf(prompt)

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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    except Exception as e:
        print(f"⚠️ LLaMA 3 failed to load or generate: {e}\nFalling back to GPT-4o...")
        ask_openai(prompt, [])

def ask_openai(question, context):
    if not OPENAI_API_KEY:
        print("\u274C OPENAI_API_KEY not set. Cannot use OpenAI fallback.")
        return
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    prompt = generate_prompt(question, context)
    print(f"\n\U0001F4AC Answer (OpenAI {openai_model_name}):\n")
    response = llm.invoke(prompt)
    print(response.content)

def ask_question(question, model_choice="mixtral", filter_tag=None, filter_filename=None, filter_all=False, stream=False, history=None):
    db = load_vectorstore()
    print("\n\U0001F50D Question:", question)

    filter_kwargs = {}
    if filter_tag:
        filter_kwargs["metadata"] = {"tag": filter_tag}
    if filter_filename:
        if "metadata" in filter_kwargs:
            filter_kwargs["metadata"]["source"] = filter_filename
        else:
            filter_kwargs["metadata"] = {"source": filter_filename}

    if filter_all:
        all_docs = db.similarity_search("", k=1000)
        relevant_docs = [(doc, 0.0) for doc in all_docs]
    else:
        relevant_docs = db.similarity_search_with_score(question, k=3, **filter_kwargs)

    if not relevant_docs:
        print("\u26A0\uFE0F No relevant documents found.")
        return

    docs = [doc for doc, score in relevant_docs]
    prompt = generate_prompt(question, docs)

    print("\n\U0001F4DA Top Retrieved Documents with Scores:")
    for i, (doc, score) in enumerate(relevant_docs):
        snippet = doc.page_content[:120].replace("\n", " ")
        source = doc.metadata.get("source", "N/A")
        tag = doc.metadata.get("tag", "N/A")
        print(f"[{i+1}] Score: {float(score):.4f} | File: {source} | Tag: {tag} | {snippet}...")

    doc_by_source = defaultdict(list)
    for doc, _ in relevant_docs:
        source = doc.metadata.get("source", "unknown")
        doc_by_source[source].append(doc)

    print("\n📁 Document Contribution Summary:")
    for src, doc_list in doc_by_source.items():
        print(f"- {src}: {len(doc_list)} chunk(s)")

    chunk_metadata = []
    for i, (doc, score) in enumerate(relevant_docs):
        chunk_metadata.append({
            "rank": int(i + 1),
            "score": float(score),
            "source": str(doc.metadata.get("source", "N/A")),
            "tag": str(doc.metadata.get("tag", "N/A")),
            "content": doc.page_content[:300]
        })

    chunk_db_path = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "chunk_db.json")
    with open(chunk_db_path, "w") as f:
        json.dump(chunk_metadata, f, indent=2)
    print(f"\n📦 Exported chunk metadata to: {chunk_db_path}")

    if model_choice == "gpt4o":
        ask_openai(question, docs)
    elif model_choice == "llama3":
        ask_llama3_hf(prompt)
    elif model_choice == "mixtral":
        ask_mistral_gguf(prompt, stream=stream)
        history = conversation_history.copy() if stream else []
        ask_mistral_gguf(prompt, stream=stream, history=history)
        if stream:
            conversation_history[:] = history
    else:
        raise ValueError("Invalid model choice: must be one of ['mixtral', 'llama3', 'gpt4o']")

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
    args = parser.parse_args()

    ask_question(
        question=args.question,
        model_choice=args.model,
        filter_tag=args.filter_tag,
        filter_filename=args.filter_file,
        filter_all=args.filter_all,
        stream=args.stream
        )
    if args.chat:
        ask_question(
            question=args.question,
            model_choice=args.model,
            filter_tag=args.filter_tag,
            filter_filename=args.filter_file,
            filter_all=args.filter_all,
            stream=args.stream,
            history=conversation_history
        )
    else:
        ask_question(
            question=args.question,
            model_choice=args.model,
            filter_tag=args.filter_tag,
            filter_filename=args.filter_file,
            filter_all=args.filter_all,
            stream=args.stream
        )