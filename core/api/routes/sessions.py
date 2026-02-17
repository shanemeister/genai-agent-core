"""Session persistence endpoints: save, load, list, delete, export."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.artifacts.storage_sessions_pg import (
    ChatMessage as SessionChatMessage,
    ChatSession,
    save_session,
    load_session,
    list_sessions,
    delete_session,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SaveSessionRequest(BaseModel):
    session_id: str
    title: str
    messages: List[dict]


@router.post("/save")
async def save_chat_session(req: SaveSessionRequest):
    """Save or update a chat session."""
    messages = [SessionChatMessage.model_validate(m) for m in req.messages]
    session = ChatSession(
        id=req.session_id,
        title=req.title,
        messages=messages,
    )
    # If loading an existing session, preserve its created_at
    existing = await load_session(req.session_id)
    if existing:
        session.created_at = existing.created_at
    await save_session(session)
    return {"status": "saved", "session_id": session.id, "message_count": len(messages)}


@router.get("", response_model=None)
async def get_sessions_list(limit: int = Query(default=50, ge=1, le=200)):
    """List all sessions (summaries without full messages)."""
    return await list_sessions(limit=limit)


@router.get("/{session_id}")
async def get_chat_session(session_id: str):
    """Load a chat session by ID."""
    session = await load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump(mode="json")


@router.delete("/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.get("/{session_id}/export")
async def export_session_markdown(session_id: str):
    """Export a chat session as markdown."""
    session = await load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    lines: list[str] = []
    lines.append(f"# {session.title}")
    lines.append("")
    lines.append(f"**Session:** {session.id}")
    lines.append(f"**Created:** {session.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Messages:** {len(session.messages)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in session.messages:
        if msg.role == "user":
            lines.append("## You")
        else:
            header = "## Assistant"
            meta_parts = []
            if msg.model:
                meta_parts.append(msg.model)
            if msg.processing_time:
                meta_parts.append(f"{msg.processing_time:.1f}s")
            if meta_parts:
                header += f" ({' | '.join(meta_parts)})"
            lines.append(header)

        lines.append("")
        lines.append(msg.text)
        lines.append("")

        # Include reasoning if present
        if msg.reasoning:
            lines.append("<details>")
            lines.append("<summary>Reasoning (chain-of-thought)</summary>")
            lines.append("")
            lines.append(msg.reasoning)
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Include retrieved context
        if msg.retrieved_context:
            lines.append("<details>")
            lines.append(f"<summary>Retrieved context ({len(msg.retrieved_context)} sources)</summary>")
            lines.append("")
            for doc in msg.retrieved_context:
                doc_type = "Memory Card" if doc.get("doc_id", "").startswith("memory:") else "Seed Doc"
                lines.append(f"- **{doc_type}** (score: {doc.get('score', 0):.3f}): {doc.get('text', '')}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Include proposed memories
        if msg.proposed_memories:
            lines.append(f"**Proposed memories ({len(msg.proposed_memories)}):**")
            lines.append("")
            for pm in msg.proposed_memories:
                status = pm.get("approval", "pending")
                icon = "+" if status == "approved" else "-" if status == "rejected" else "?"
                lines.append(f"- [{icon}] {pm.get('text', '')} *({pm.get('category', '')})*")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append(f"*Exported from Noesis on {session.updated_at.strftime('%Y-%m-%d %H:%M UTC')}*")

    markdown = "\n".join(lines)
    return {
        "session_id": session_id,
        "title": session.title,
        "markdown": markdown,
        "filename": f"{session.title.replace(' ', '_')[:40]}_{session_id}.md",
    }
