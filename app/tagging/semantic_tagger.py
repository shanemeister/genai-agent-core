# semantic_tagger.py

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

DATA_DIR = "../data"
VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_tag_documents() -> List[Document]:
    print("📁 Scanning PDFs for semantic tagging...")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    tagger = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    topics = ["resume", "cover letter", "AI research", "founder pitch", "technical report"]
    tagged_docs = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_DIR, pdf_file))
        chunks = loader.load_and_split()

        for chunk in chunks:
            prediction = tagger(chunk.page_content, topics)
            top_label = prediction["labels"][0]
            chunk.metadata["source"] = pdf_file
            chunk.metadata["topic_tag"] = top_label
            tagged_docs.append(chunk)
            print(f"🔖 Tagged chunk from {pdf_file} as '{top_label}'")

    print(f"✅ Total tagged chunks: {len(tagged_docs)}")
    return tagged_docs


def embed_and_store(documents: List[Document]):
    print("🔍 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("📦 Embedding and saving to FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("✅ FAISS index with tags saved.")


if __name__ == "__main__":
    docs = load_and_tag_documents()
    embed_and_store(docs)