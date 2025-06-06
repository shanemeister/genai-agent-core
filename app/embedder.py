import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_DIR = Path("data/")
VECTORSTORE_DIR = Path("vectorstore/")

def load_and_chunk_pdfs(data_path=DATA_DIR):
    all_chunks = []
    pdf_files = list(data_path.glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF(s)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

def embed_chunks(chunks):
    print("🔍 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("📦 Embedding and indexing...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"✅ FAISS index saved to {VECTORSTORE_DIR}")

if __name__ == "__main__":
    chunks = load_and_chunk_pdfs()
    embed_chunks(chunks)