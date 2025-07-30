from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.interface.query_handler import ask_question
from chat.vectorstore_memory import rebuild_vectorstore_from_documents

router = APIRouter()

class AskRequest(BaseModel):
    question: str = Field(..., example="What is RAG?", description="The user query or question.")
    session_id: Optional[str] = Field(None, description="Session ID to track conversational history")
    model: Optional[str] = Field("mixtral", description="Model to use: mixtral, llama3, or gpt4o")
    filter_tag: Optional[str] = Field(None, description="Filter documents by semantic tag")
    filter_filename: Optional[str] = Field(None, description="Restrict search to a specific filename")
    filter_all: Optional[bool] = Field(False, description="Use all documents (no semantic filtering)")
    stream: Optional[bool] = Field(False, description="Stream the response if supported by the model")
    chat_enabled: Optional[bool] = Field(True, description="Enable persistent chat memory")

class AskResponse(BaseModel):
    answer: str
    meta: Dict[str, Any]

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question",
    description="Ask a question using a selected model and document context."
)
def ask_endpoint(req: AskRequest):
    """
    Ask a question using a selected model and document context.
    """
    try:
        response = ask_question(
            question=req.question,
            model_choice=req.model,
            filter_tag=req.filter_tag,
            filter_filename=req.filter_filename,
            filter_all=req.filter_all,
            stream=req.stream,
            session_id=req.session_id,
            chat_enabled=req.chat_enabled
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RebuildResponse(BaseModel):
    status: str

@router.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Rebuild the vectorstore",
    description="Rebuild the vectorstore from documents."
)
def rebuild_endpoint():
    """
    Rebuild the vectorstore from documents.
    """
    try:
        rebuild_vectorstore_from_documents()
        return {"status": "✅ Vectorstore rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
