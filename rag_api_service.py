from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from app.interface.query_handler import ask_question
from chat.vectorstore_memory import rebuild_vectorstore_from_documents
from fastapi import FastAPI
from app.api import routes

router = APIRouter()

class AskRequest(BaseModel):
    query: str = Field(..., example="What is retrieval-augmented generation?")
    model: Optional[str] = Field("mixtral", example="gpt4o")
    session_id: Optional[str] = None
    chat_enabled: Optional[bool] = True
    filter_tag: Optional[str] = None
    filter_filename: Optional[str] = None
    filter_all: Optional[bool] = False
    stream: Optional[bool] = False

@router.post("/ask")
def ask_route(req: AskRequest):
    """
    Answer a question using the selected LLM and optional filters.
    """
    response = ask_question(
        question=req.query,
        model_choice=req.model,
        filter_tag=req.filter_tag,
        filter_filename=req.filter_filename,
        filter_all=req.filter_all,
        stream=req.stream,
        session_id=req.session_id,
        chat_enabled=req.chat_enabled,
    )
    return {
        "answer": response["answer"],
        "meta": response["meta"]
    }

@router.post("/rebuild")
def rebuild_vectorstore():
    """
    Rebuild the FAISS vectorstore from current documents.
    """
    try:
        rebuild_vectorstore_from_documents()
        return {"status": "✅ Vectorstore rebuilt successfully"}
    except Exception as e:
        return {"error": str(e)}

app = FastAPI(
title="GenAI Agent Core API",
description="RAG + Agent API for local and OpenAI LLMs",
version="0.1.0"
)

app.include_router(routes.router)