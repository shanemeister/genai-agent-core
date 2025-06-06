import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import torch

VECTORSTORE_PATH = "vectorstore"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
MODEL_DIR = "models/mistral-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(VECTORSTORE_PATH, embeddings)

def generate_prompt(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    return (
        f"You are a helpful assistant. Use the context below to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
        use_safetensors=True,
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir="models/mistral-7b",
        trust_remote_code=True,
        use_safetensors=True,
        token=HF_TOKEN
    )
    return tokenizer, model.to(DEVICE)

def ask_question(question):
    tokenizer, model = load_model_and_tokenizer()
    db = load_vectorstore()
    relevant_docs = db.similarity_search(question, k=3)
    if not relevant_docs:
        print("⚠️ No relevant documents found.")
        return
    prompt = generate_prompt(question, relevant_docs)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🔍 Question:", question)
    print("\n💬 Answer:", answer)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query.py 'Your question here'")
    else:
        ask_question(sys.argv[1])