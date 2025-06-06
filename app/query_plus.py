import os
import sys
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI


VECTORSTORE_PATH = "vectorstore"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
SAVE_PATH = "models/mistral-7b"
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
MODEL_DIR = "models/mistral-7b"  # Load from locally stored model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with open("llm_config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name = config["models"]["gpt_4o_mini"]["model"]

llm = ChatOpenAI(model=model_name, temperature=0.3)

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

def ask_local_llm(prompt):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
        use_safetensors=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
        use_safetensors=True
    )

    tokenizer.save_pretrained(SAVE_PATH)
    # model.save_pretrained(SAVE_PATH) # Uncomment if you want to save the model weights
    
    print(f"✅ Model and tokenizer saved to {SAVE_PATH}")
    model = model.to(DEVICE)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n💬 Answer (Mistral - Local):\n")
    model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        generation_config=None  # 👈 disables internal defaults
    )

def ask_openai(question, context):
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set. Cannot use OpenAI fallback.")
        return
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    prompt = generate_prompt(question, context)
    print(f"\n💬 Answer (OpenAI {model_name}):\n")
    response = llm.invoke(prompt)
    print(response.content)

def ask_question(question, use_openai=False):
    db = load_vectorstore()
    relevant_docs = db.similarity_search_with_score(question, k=3)

    if not relevant_docs:
        print("⚠️ No relevant documents found.")
        return

    docs = [doc for doc, score in relevant_docs]
    prompt = generate_prompt(question, docs)

    print("\n🔍 Question:", question)
    print("\n📚 Top Retrieved Documents with Scores:")
    for i, (doc, score) in enumerate(relevant_docs):
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"[{i+1}] Score: {score:.4f} | {snippet}...")

    if use_openai:
        ask_openai(question, docs)
    else:
        ask_local_llm(prompt)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_plus.py 'Your question here' [--openai]")
    else:
        q = sys.argv[1]
        use_openai = len(sys.argv) > 2 and sys.argv[2] == "--openai"
        ask_question(q, use_openai)