import os
import glob
import yaml
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

DATA_DIR = "data"
VECTORSTORE_PATH = "vectorstore"
TAGS_FILE = "pdf_tags.yaml"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_tags():
    if os.path.exists(TAGS_FILE):
        with open(TAGS_FILE, "r") as f:
            return yaml.safe_load(f)
    return {}


def chunk_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    filename = os.path.basename(path)
    return [
        Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "source": filename,
                "tags": tag_lookup.get(filename, [])
            },
        )
        for doc in docs
    ]


def embed_documents(all_docs):
    print("\n✅ Total chunks created:", len(all_docs))
    print("🔍 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("📦 Embedding and indexing...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("✅ FAISS index saved to", VECTORSTORE_PATH)


def main():
    print("📁 Scanning PDFs in data directory...")
    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    all_docs = []
    for path in tqdm(pdf_paths, desc="📄 Processing PDFs"):
        all_docs.extend(chunk_pdf(path))
    embed_documents(all_docs)


if __name__ == "__main__":
    tag_lookup = load_tags()
    main()