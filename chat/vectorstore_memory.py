from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import List
import fitz  # PyMuPDF
import docx  # python-docx

# embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTOR_DB_PATH = "vectorstore"

def get_vectorstore():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True)
    return FAISS.from_texts([], embedding)

def retrieve_similar_context(query: str, k=3):
    vs = get_vectorstore()
    results = vs.similarity_search(query, k=k)
    return [r.page_content for r in results]

def add_to_vectorstore(texts: list[str]):
    vs = get_vectorstore()
    vs.add_texts(texts)
    vs.save_local(VECTOR_DB_PATH)

def rebuild_vectorstore_from_directory(directory_path="docs"):
    if not os.path.isdir(directory_path):
        print(f"❌ Directory '{directory_path}' does not exist.")
        return

    text_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".txt")]
    if not text_files:
        print(f"⚠️ No .txt files found in '{directory_path}'.")
        return

    texts = []
    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except Exception as e:
            print(f"⚠️ Failed to read {file_path}: {e}")

    if texts:
        vs = FAISS.from_texts(texts, embedding)
        vs.save_local(VECTOR_DB_PATH)
        print(f"✅ Rebuilt vectorstore from {len(texts)} file(s).")
    else:
        print("⚠️ No valid content to embed.")
    
    
def extract_text_from_pdf(filepath: str) -> str:
    try:
        with fitz.open(filepath) as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"❌ Error reading PDF {filepath}: {e}")
        return ""

def extract_text_from_docx(filepath: str) -> str:
    try:
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"❌ Error reading DOCX {filepath}: {e}")
        return ""

def rebuild_vectorstore_from_documents(base_dir="data") -> None:
    supported_ext = {".txt", ".pdf", ".docx"}
    texts: List[str] = []
    metadata: List[dict] = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in supported_ext:
                continue

            full_path = os.path.join(root, file)
            rel_folder = os.path.relpath(root, base_dir)

            try:
                if ext == ".txt":
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                elif ext == ".pdf":
                    content = extract_text_from_pdf(full_path)
                elif ext == ".docx":
                    content = extract_text_from_docx(full_path)
                else:
                    continue  # unreachable
                if content.strip():
                    texts.append(content)
                    metadata.append({
                        "source": file,
                        "tag": rel_folder
                    })
            except Exception as e:
                print(f"⚠️ Failed to process {full_path}: {e}")

    # Placeholder for Kindle support in the future

    if texts:
        vs = FAISS.from_texts(texts, embedding, metadatas=metadata)
        vs.save_local(VECTOR_DB_PATH)
        print(f"✅ Rebuilt vectorstore from {len(texts)} document(s).")
    else:
        print("⚠️ No valid documents found to embed.")
        