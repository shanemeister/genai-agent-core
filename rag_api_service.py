from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from chat.vectorstore_memory import retrieve_similar_context
from app.interface.query_plus import ask_question
from chat.vectorstore_memory import rebuild_vectorstore_from_documents

app = FastAPI()

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is retrieval-augmented generation?")
    session_id: Optional[str] = Field(None, example="user123")
    model: Optional[str] = Field("mixtral", description="LLM model to use", example="gpt4o")
    chat_enabled: Optional[bool] = Field(True, description="Enable chat memory")
    filter_tag: Optional[str] = Field(None, example="interview")
    filter_filename: Optional[str] = Field(None, example="my_notes.pdf")
    filter_all: Optional[bool] = Field(False, description="Use all documents")
    stream: Optional[bool] = Field(False, description="Enable streaming (Mixtral only)")

import time

@app.post("/ask")
def ask(req: QueryRequest):
    try:
        print(f"🔍 Incoming query: {req.query} (session_id={req.session_id})")
        start_time = time.time()

        # vectorstore retrieval for metadata
        from chat.vectorstore_memory import get_vectorstore
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

        # run the model
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

        elapsed = round(time.time() - start_time, 2)

        return {
            "answer": response["answer"],
            "session_id": req.session_id,
            "sources": chunks,
            "meta": response.get("meta", {})
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