from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from chat.vectorstore_memory import retrieve_similar_context, get_vectorstore, rebuild_vectorstore_from_documents
from app.interface.query_plus import ask_question, LLAMA3_MODEL_PATH
import time

app = FastAPI()

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is retrieval-augmented generation?")
    session_id: Optional[str] = Field(None, example="demo1")
    model: Optional[str] = Field("mixtral", example="llama3")
    chat_enabled: Optional[bool] = Field(True)
    filter_tag: Optional[str] = None
    filter_filename: Optional[str] = None
    filter_all: Optional[bool] = Field(False)
    stream: Optional[bool] = Field(False)
    token_estimate: Optional[bool] = Field(False, description="Estimate token usage (for local models)")

@app.post("/ask")
def ask(req: QueryRequest):
    try:
        print(f"🔍 Incoming query: {req.query} (session_id={req.session_id})")
        start_time = time.time()

        vs = get_vectorstore()
        results = vs.similarity_search_with_score(req.query, k=3)
        chunks = [
            {
                "score": float(score),
                "source": doc.metadata.get("source", "N/A"),
                "tag": doc.metadata.get("tag", "N/A"),
                "content": doc.page_content[:300]
            }
            for doc, score in results
        ]

        response = ask_question(
            question=req.query,
            model_choice=req.model,
            filter_tag=req.filter_tag,
            filter_filename=req.filter_filename,
            filter_all=req.filter_all,
            stream=req.stream,
            session_id=req.session_id,
            chat_enabled=req.chat_enabled,
            history=None
        )

        token_count = None
        if req.token_estimate and req.model in ["llama3", "mixtral"]:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                LLAMA3_MODEL_PATH if req.model == "llama3" else "gpt2",
                local_files_only=True
            )
            token_count = len(tokenizer.encode(req.query))

        elapsed = round(time.time() - start_time, 2)

        return {
            "answer": response["answer"],
            "session_id": req.session_id,
            "sources": chunks,
            "meta": {
                **response.get("meta", {}),
                "tokens_estimated": token_count,
                "model": req.model,
                "elapsed_seconds": elapsed
            }
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/rebuild")
def rebuild_vectorstore():
    try:
        rebuild_vectorstore_from_documents()
        return {"status": "✅ Vectorstore rebuilt successfully"}
    except Exception as e:
        return {"error": str(e)}