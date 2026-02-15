from __future__ import annotations
import json as json_mod
import os
import re
import time
import uuid
from core.artifacts.storage_sqlite import load_all_cards, upsert_card
from core.artifacts.storage_sessions import (
    ChatMessage as SessionChatMessage,
    ChatSession,
    save_session,
    load_session,
    list_sessions,
    delete_session,
    init_sessions_db,
)
from core.artifacts.mindfile_entry import MindFileEntry, MindFileEntryCategory
from core.artifacts.storage_mindfile import (
    init_mindfile_db,
    save_entry as save_mindfile_entry,
    load_all_entries as load_mindfile_entries,
    load_entry as load_mindfile_entry,
    update_note as update_mindfile_note,
    delete_entry as delete_mindfile_entry,
    entry_exists_for_card,
)
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import httpx

from core.validation.models import ValidationResult

from core.artifacts.memory_card import (
    MemoryApproval,
    MemoryCard,
    MemoryCategory,
    MemoryProvenance,
    MemoryScope,
)
from core.graph.models import (
    ConceptCreate,
    DiagramToGraphRequest,
    GraphData,
    GraphEdge,
    GraphNode,
    GraphToDiagramRequest,
    NeighborRequest,
    RelationshipCreate,
)
from core.graph.neo4j_client import close_driver, init_driver
from core.graph.schema import ensure_schema
from core.graph import queries as graph_queries
from fastapi.middleware.cors import CORSMiddleware

# LLM Configuration
LLM_ENDPOINT = "http://127.0.0.1:8080/ask"
LLM_TIMEOUT = 180.0  # DeepSeek-R1 reasoning can take 60-120s on complex questions

async def ask_llm(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> dict:
    """Call the local LLM service and return full response."""
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        response = await client.post(
            LLM_ENDPOINT,
            json={
                "question": question,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        response.raise_for_status()
        return response.json()

app = FastAPI(title="Noesis API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# V0: in-memory store (swap to SQLite later)
MEMORY_CARDS: dict[str, MemoryCard] = {}

@app.on_event("startup")
async def _startup():
    global MEMORY_CARDS
    MEMORY_CARDS = await load_all_cards()
    await init_sessions_db()
    await init_mindfile_db()
    try:
        await init_driver()
        await ensure_schema()
    except Exception as e:
        print(f"[WARN] Neo4j not available at startup: {e}")


@app.on_event("shutdown")
async def _shutdown():
    await close_driver()


class ProposeMemoryRequest(BaseModel):
    text: str
    category: MemoryCategory
    scope: MemoryScope = MemoryScope.PROJECT
    reason: str
    derived_from_artifact_ids: List[str] = []
    tools_used: List[str] = []
    model: Optional[str] = None

@app.get("/llm/test")
async def test_llm():
    try:
        result = await ask_llm("Say 'Hello from the LLM!'", max_tokens=50)
        return {"status": "ok", "response": result["answer"], "model": result.get("model"), "processing_time": result.get("processing_time")}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/memory/propose", response_model=MemoryCard)
async def propose_memory(req: ProposeMemoryRequest):
    card = MemoryCard(
        text=req.text.strip(),
        category=req.category,
        scope=req.scope,
        provenance=MemoryProvenance(
            reason=req.reason.strip(),
            derived_from_artifact_ids=req.derived_from_artifact_ids,
            tools_used=req.tools_used,
            model=req.model,
            sources=[],
        ),
    )
    MEMORY_CARDS[card.id] = card
    await upsert_card(card)
    return card


@app.get("/memory/cards", response_model=List[MemoryCard])
def list_cards(
    approval: Optional[MemoryApproval] = Query(default=None),
    category: Optional[MemoryCategory] = Query(default=None),
    q: Optional[str] = Query(default=None, description="substring search"),
):
    items = list(MEMORY_CARDS.values())

    if approval:
        items = [c for c in items if c.approval == approval]
    if category:
        items = [c for c in items if c.category == category]
    if q:
        needle = q.lower()
        items = [c for c in items if needle in c.text.lower() or needle in c.provenance.reason.lower()]

    # newest first
    items.sort(key=lambda c: c.created_at, reverse=True)
    return items


@app.post("/memory/cards/{card_id}/approve", response_model=MemoryCard)
async def approve_card(card_id: str):
    card = MEMORY_CARDS.get(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    if card.approval != MemoryApproval.PENDING:
        return card
    card.approval = MemoryApproval.APPROVED
    from datetime import datetime
    card.approved_at = datetime.utcnow()
    MEMORY_CARDS[card_id] = card
    await upsert_card(card)
    # Auto-sync approved card to knowledge graph
    try:
        await graph_queries.sync_memory_card(card)
    except Exception:
        pass  # Graph sync is best-effort
    # Auto-promote to Mind File
    try:
        if not await entry_exists_for_card(card.id):
            entry = MindFileEntry(
                category=MindFileEntryCategory(card.category.value),
                text=card.text,
                source_memory_card_id=card.id,
            )
            await save_mindfile_entry(entry)
            # Sync Mind File entry to Neo4j graph
            try:
                await graph_queries.sync_mindfile_entry(entry)
            except Exception:
                pass  # Graph sync is best-effort
    except Exception:
        pass  # Mind File promotion is best-effort
    return card


@app.post("/memory/cards/{card_id}/reject", response_model=MemoryCard)
async def reject_card(card_id: str):
    card = MEMORY_CARDS.get(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    if card.approval != MemoryApproval.PENDING:
        return card
    card.approval = MemoryApproval.REJECTED
    from datetime import datetime
    card.rejected_at = datetime.utcnow()
    MEMORY_CARDS[card.id] = card
    await upsert_card(card)
    return card


# Seed two “values/framing” cards (your requested statements) for V0 demo
# Seed two “values/framing” cards (your requested statements) for V0 demo
@app.post("/dev/seed")
async def dev_seed():
    seeds = [
        (
            "A system that cannot recognize values worth dying for cannot recognize values worth preserving.",
            MemoryCategory.PRINCIPLES_VALUES,
            "User-approved value anchor; informs ethical boundaries and system behavior.",
        ),
        (
            "Nothing is new in the human mind, but our tools are sharper than ever.",
            MemoryCategory.COGNITIVE_FRAMING,
            "User-approved framing principle; emphasizes tool power vs human novelty.",
        ),
    ]
    created = []
    skipped = []

    for text, cat, reason in seeds:
        # Check if a card with this exact text already exists
        existing = [c for c in MEMORY_CARDS.values() if c.text == text]
        if existing:
            skipped.append({"text": text[:50] + "...", "reason": "already exists"})
            continue

        card = MemoryCard(
            text=text,
            category=cat,
            scope=MemoryScope.PROJECT,
            provenance=MemoryProvenance(
                reason=reason,
                derived_from_artifact_ids=[],
                tools_used=["dev_seed"],
                model=None,
                sources=[],
            ),
        )
        MEMORY_CARDS[card.id] = card
        await upsert_card(card)   # <-- persist to SQLite
        created.append(card.id)

    return {"seeded": created, "count": len(created), "skipped": skipped}


# ============================================================================
# Chat & Artifact Generation Endpoints
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 2000

class RetrievedDoc(BaseModel):
    doc_id: str
    text: str
    score: float

class GroundingScore(BaseModel):
    """How well-grounded a response is in the knowledge base."""
    overall: float  # 0.0–1.0 composite score
    retrieval: float  # avg relevance of retrieved docs
    coverage: float  # % of query concepts found in knowledge graph
    diversity: float  # source diversity (0=one source, 1=many independent sources)
    reasoning: float  # 0 or 1 — did the model reason?
    label: str  # "High", "Medium", "Low", "Ungrounded"
    detail: str  # human-readable summary

class ChatResponse(BaseModel):
    session_id: str
    response: str
    reasoning: Optional[str] = None
    model: Optional[str] = None
    processing_time: Optional[float] = None
    retrieved_context: List[RetrievedDoc] = []
    proposed_memories: List[MemoryCard] = []
    proposed_diagram: Optional[str] = None
    grounding: Optional[GroundingScore] = None
    validation: Optional[ValidationResult] = None


def _split_thinking(raw: str) -> tuple[str, Optional[str]]:
    """Separate reasoning from the visible answer.

    DeepSeek-R1 models emit chain-of-thought in several formats:
    1. <think>reasoning</think>answer  (full tags)
    2. reasoning</think>answer  (opening tag stripped by vLLM chat template)
    Returns (answer, reasoning) where reasoning may be None.
    """
    # Case 1: Full <think>...</think> tags
    match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        answer = raw[: match.start()] + raw[match.end() :]
        # Handle multiple think blocks
        while "<think>" in answer:
            m2 = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
            if not m2:
                break
            reasoning += "\n\n" + m2.group(1).strip()
            answer = answer[: m2.start()] + answer[m2.end() :]
        return answer.strip(), reasoning if reasoning else None

    # Case 2: Opening <think> stripped by vLLM — text ends with </think>
    if "</think>" in raw:
        parts = raw.split("</think>", 1)
        reasoning = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return answer, reasoning if reasoning else None

    return raw.strip(), None


async def _compute_grounding(
    query: str,
    context_docs: list[dict],
    has_reasoning: bool,
) -> GroundingScore:
    """Compute a grounding score for a chat response.

    Dimensions:
      retrieval  — average relevance score of retrieved docs (0-1)
      coverage   — fraction of query concepts that exist in the knowledge graph
      diversity  — how many independent sources contributed (0=none, 1=many)
      reasoning  — binary: did the model engage in chain-of-thought?
    """
    # --- Retrieval quality ---
    # Scores may be cosine similarity (0-1) or dot products (100+).
    # Normalize to 0-1: use top score as reference. If scores are already
    # in 0-1, use directly. If large, map relative to best score.
    raw_scores = [d["score"] for d in context_docs if d.get("score")]
    if raw_scores:
        max_s = max(raw_scores)
        if max_s > 1.5:
            # Dot-product / cross-encoder scores — normalize relative to top
            # Top score maps to 1.0, lower scores scale proportionally
            scores = [s / max_s for s in raw_scores]
        else:
            scores = [max(0.0, min(1.0, s)) for s in raw_scores]
        retrieval = sum(scores) / len(scores)
    else:
        retrieval = 0.0

    # --- Coverage: check how many query concepts exist in Neo4j ---
    coverage = 0.0
    try:
        from core.graph.concept_extractor import extract_concepts
        query_concepts = extract_concepts(query, max_concepts=5)
        if query_concepts:
            from core.graph import queries as gq
            found = 0
            for concept_name in query_concepts:
                results = await gq.search_nodes(concept_name, limit=1)
                if results.nodes:
                    found += 1
            coverage = found / len(query_concepts)
    except Exception:
        coverage = 0.0

    # --- Source diversity ---
    # Count unique source types (memory cards vs seed docs)
    unique_sources = set()
    for d in context_docs:
        doc_id = d.get("doc_id", "")
        if doc_id.startswith("memory:"):
            unique_sources.add(doc_id)  # each card is independent
        else:
            unique_sources.add("seed")  # all seed docs group together
    n = len(unique_sources)
    # Map to 0-1: 0 sources=0, 1 source=0.25, 2=0.5, 3+=0.75, 5+=1.0
    diversity = min(n / 4.0, 1.0) if n > 0 else 0.0

    # --- Reasoning ---
    reasoning_score = 1.0 if has_reasoning else 0.0

    # --- Composite ---
    # Weighted: retrieval 40%, coverage 30%, diversity 20%, reasoning 10%
    overall = (
        0.40 * retrieval
        + 0.30 * coverage
        + 0.20 * diversity
        + 0.10 * reasoning_score
    )
    overall = round(overall, 3)

    # Label
    if overall >= 0.7:
        label = "High"
    elif overall >= 0.4:
        label = "Medium"
    elif overall > 0.1:
        label = "Low"
    else:
        label = "Ungrounded"

    # Detail
    parts = []
    if retrieval > 0:
        parts.append(f"retrieval relevance {retrieval:.0%}")
    else:
        parts.append("no relevant context found")
    if coverage > 0:
        parts.append(f"{coverage:.0%} concepts in knowledge graph")
    else:
        parts.append("no prior knowledge of these concepts")
    if n > 1:
        parts.append(f"{n} independent sources")
    elif n == 1:
        parts.append("single source")
    if has_reasoning:
        parts.append("model used chain-of-thought")

    detail = "; ".join(parts)

    return GroundingScore(
        overall=overall,
        retrieval=round(retrieval, 3),
        coverage=round(coverage, 3),
        diversity=round(diversity, 3),
        reasoning=reasoning_score,
        label=label,
        detail=detail,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint - implements the core Ask → Retrieve → Reason → Respond loop.
    """
    # 1. Retrieve context from RAG (seed docs + approved memory cards)
    from core.rag.retriever import retrieve_context, index_memory_cards
    index_memory_cards(
        [c for c in MEMORY_CARDS.values() if c.approval == MemoryApproval.APPROVED]
    )
    context_docs = retrieve_context(req.message, k=5)
    context_text = "\n".join([f"- {doc['text']}" for doc in context_docs])

    # 2. Build prompt with context
    prompt = _build_chat_prompt(req.message, context_text)

    # 3. Call LLM
    llm_response = await ask_llm(
        question=prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )

    # 3b. Separate <think> reasoning from visible answer
    raw_answer = llm_response["answer"]
    visible_answer, reasoning = _split_thinking(raw_answer)

    # 4. Build retrieved context for frontend display
    retrieved = [
        RetrievedDoc(doc_id=d["doc_id"], text=d["text"], score=round(d["score"], 3))
        for d in context_docs
    ]

    # 5. Record lineage in Neo4j (best-effort)
    session_id = f"chat-{uuid.uuid4().hex[:12]}"
    try:
        await graph_queries.record_chat_session(
            session_id=session_id,
            user_message=req.message,
            assistant_response=visible_answer,
            model=llm_response.get("model"),
            processing_time=llm_response.get("processing_time"),
            retrieved_doc_ids=[d["doc_id"] for d in context_docs],
        )
    except Exception:
        pass

    # 6. Auto-propose memory cards (best-effort, non-blocking)
    proposed_cards = await _propose_memories_for_session(
        req.message, visible_answer, session_id, llm_response.get("model")
    )

    # 7. Grounding score
    grounding = None
    try:
        grounding = await _compute_grounding(
            query=req.message,
            context_docs=context_docs,
            has_reasoning=reasoning is not None,
        )
    except Exception:
        pass

    return ChatResponse(
        session_id=session_id,
        response=visible_answer,
        reasoning=reasoning,
        model=llm_response.get("model"),
        processing_time=llm_response.get("processing_time"),
        retrieved_context=retrieved,
        proposed_memories=proposed_cards,
        grounding=grounding,
    )


def _build_chat_prompt(user_message: str, context_text: str) -> str:
    """Build the full system + context + user prompt."""
    return f"""You are Noesis, a private AI assistant for knowledge organization and reasoning. You run locally and prioritize accuracy over helpfulness.

CRITICAL RULES:
- Only state facts you are confident about. If you are uncertain or the topic is outside your training data, say "I'm not confident about this" and explain what you do know vs. what you're unsure about.
- Never fabricate definitions, dates, or technical details. Getting something wrong is worse than admitting uncertainty.
- Your training data has a cutoff date. If asked about recent events, products, or updates, acknowledge that your information may be outdated.
- Use the retrieved context below when relevant, but don't force it into answers where it doesn't apply.
- If a diagram or graph would help explain your response, mention it.

Retrieved context from the knowledge base:
{context_text}

User: {user_message}"""


async def _propose_memories_for_session(
    user_message: str,
    visible_answer: str,
    session_id: str,
    model: str | None,
) -> List[MemoryCard]:
    """Auto-propose memory cards (best-effort)."""
    proposed_cards: List[MemoryCard] = []
    try:
        from core.artifacts.memory_proposer import propose_memories
        proposals = await propose_memories(
            user_message=user_message,
            assistant_response=visible_answer,
        )
        for p in proposals:
            card = MemoryCard(
                text=p["text"],
                category=MemoryCategory(p["category"]),
                scope=MemoryScope.PROJECT,
                provenance=MemoryProvenance(
                    reason=p["reason"],
                    derived_from_artifact_ids=[session_id],
                    tools_used=["auto_propose"],
                    model=model,
                    sources=[],
                ),
            )
            MEMORY_CARDS[card.id] = card
            await upsert_card(card)
            proposed_cards.append(card)
            try:
                await graph_queries.link_memory_to_session(card.id, session_id)
            except Exception:
                pass
    except Exception:
        pass
    return proposed_cards


# ---------------------------------------------------------------------------
# Streaming chat endpoint — calls vLLM directly (bypasses main_vllm.py proxy)
# ---------------------------------------------------------------------------
VLLM_BASE = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8081")
VLLM_MODEL = os.getenv("VLLM_MODEL_NAME", "./models/deepseek-r1-70b-w4a16")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming chat — tokens appear in real-time.

    Event types sent to the client:
      event: status    data: {"phase": "retrieving"}
      event: status    data: {"phase": "thinking"}   — DeepSeek <think> block
      event: status    data: {"phase": "answering"}   — visible answer started
      event: token     data: {"token": "..."}         — one or more chars of answer
      event: done      data: {session_id, reasoning, model, processing_time,
                              retrieved_context, proposed_memories}
      event: error     data: {"detail": "..."}
    """
    # 1. RAG retrieval (fast, ~50ms)
    from core.rag.retriever import retrieve_context, index_memory_cards
    index_memory_cards(
        [c for c in MEMORY_CARDS.values() if c.approval == MemoryApproval.APPROVED]
    )
    context_docs = retrieve_context(req.message, k=5)
    context_text = "\n".join([f"- {doc['text']}" for doc in context_docs])
    prompt = _build_chat_prompt(req.message, context_text)

    retrieved = [
        {"doc_id": d["doc_id"], "text": d["text"], "score": round(d["score"], 3)}
        for d in context_docs
    ]

    async def generate():
        start = time.time()

        # Tell frontend we're starting
        yield f"event: status\ndata: {json_mod.dumps({'phase': 'retrieving'})}\n\n"

        # 2. Stream directly from vLLM (OpenAI-compatible SSE)
        #
        # DeepSeek-R1 reasoning handling:
        # vLLM's chat template strips the opening <think> tag, so the stream
        # looks like: reasoning_text...</think>\n\nactual_answer
        # We assume ALL initial tokens are reasoning until </think> appears,
        # then switch to emitting answer tokens.
        full_text = ""
        answer_started = False
        sent_thinking_status = False

        vllm_payload = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{VLLM_BASE}/v1/chat/completions",
                    json=vllm_payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        chunk = json_mod.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if not token:
                            continue

                        full_text += token

                        if answer_started:
                            # Already past </think>, emit answer tokens
                            yield f"event: token\ndata: {json_mod.dumps({'token': token})}\n\n"
                            continue

                        # Not yet in answer mode — check for </think>
                        if "</think>" in full_text:
                            answer_started = True
                            yield f"event: status\ndata: {json_mod.dumps({'phase': 'answering'})}\n\n"
                            # Emit any answer text that arrived after </think>
                            parts = full_text.split("</think>", 1)
                            after = parts[1].lstrip() if len(parts) > 1 else ""
                            if after:
                                yield f"event: token\ndata: {json_mod.dumps({'token': after})}\n\n"
                            continue

                        # Still in reasoning phase — tell frontend once
                        if not sent_thinking_status:
                            sent_thinking_status = True
                            yield f"event: status\ndata: {json_mod.dumps({'phase': 'thinking'})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json_mod.dumps({'detail': str(e)})}\n\n"
            return

        # 3. Post-processing
        elapsed = round(time.time() - start, 2)
        visible_answer, reasoning = _split_thinking(full_text)
        session_id = f"chat-{uuid.uuid4().hex[:12]}"

        # Neo4j lineage (best-effort)
        try:
            await graph_queries.record_chat_session(
                session_id=session_id,
                user_message=req.message,
                assistant_response=visible_answer,
                model="DeepSeek-R1-70B",
                processing_time=elapsed,
                retrieved_doc_ids=[d["doc_id"] for d in context_docs],
            )
        except Exception:
            pass

        # Memory proposals (best-effort)
        proposed_cards = await _propose_memories_for_session(
            req.message, visible_answer, session_id, "DeepSeek-R1-70B"
        )
        proposed_dicts = [
            {
                "id": c.id,
                "text": c.text,
                "category": c.category.value,
                "approval": c.approval.value,
                "provenance": {"reason": c.provenance.reason},
            }
            for c in proposed_cards
        ]

        # Grounding score (best-effort)
        grounding_dict = None
        try:
            grounding = await _compute_grounding(
                query=req.message,
                context_docs=context_docs,
                has_reasoning=reasoning is not None,
            )
            grounding_dict = grounding.model_dump()
        except Exception:
            pass

        # Send final done event with all metadata
        done_payload = {
            "session_id": session_id,
            "reasoning": reasoning,
            "model": "DeepSeek-R1-70B",
            "processing_time": elapsed,
            "retrieved_context": retrieved,
            "proposed_memories": proposed_dicts,
            "grounding": grounding_dict,
        }
        yield f"event: done\ndata: {json_mod.dumps(done_payload)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Response Validation endpoint
# ---------------------------------------------------------------------------

class ValidateRequest(BaseModel):
    response_text: str
    user_question: str
    max_claims: int = 6


@app.post("/validate", response_model=ValidationResult)
async def validate_response(req: ValidateRequest):
    """Validate an LLM response by extracting claims and checking evidence."""
    from core.validation.claim_extractor import extract_claims
    from core.validation.claim_validator import validate_claims

    claims = await extract_claims(
        response_text=req.response_text,
        user_question=req.user_question,
        max_claims=req.max_claims,
    )

    if not claims:
        return ValidationResult(
            claims=[],
            summary_score=0.0,
            label="Ungrounded",
            detail="No verifiable claims extracted",
        )

    return await validate_claims(claims=claims, user_question=req.user_question)


class DiagramRequest(BaseModel):
    prompt: str
    syntax: str = "mermaid"
    temperature: float = 0.3
    source_session_id: Optional[str] = None

@app.post("/diagram/generate")
async def generate_diagram(req: DiagramRequest):
    """
    Generate Mermaid diagram syntax from a natural language prompt.
    Returns clean diagram code ready to render.
    """
    system_context = """You are a Mermaid diagram expert. Generate ONLY valid Mermaid syntax, no explanations or code fences.

IMPORTANT: Keep diagrams concise — aim for 15-25 nodes maximum. Group related items rather than listing every detail. A clear overview is better than an exhaustive list.

CRITICAL SYNTAX RULES:
- Every node MUST have an ID before its shape. Example: A[Text], B{Question?}, C([Rounded])
- Use ID[Text] for rectangles (actions/processes)
- Use ID{Text?} ONLY for decisions (yes/no questions) - must end with ?
- Use ID([Text]) for rounded rectangles (start/end) — note the double brackets
- Use -->|Label| for labeled edges
- NEVER use bare (Text) or {Text} without a node ID prefix

COMMON MISTAKES TO AVOID:
WRONG: (Start) --> A[Step]  — missing node ID before parenthesis
RIGHT: S([Start]) --> A[Step]

WRONG: B{Retrieve Data} — "Retrieve Data" is not a question
RIGHT: B[Retrieve Data] --> C{Data Found?}

Examples:

Request: "Create a flowchart of user authentication"
Response:
flowchart TD
    A[User enters credentials] --> B[Validate format]
    B --> C{Format valid?}
    C -->|Yes| D[Check database]
    C -->|No| E[Show error]
    D --> F{Credentials match?}
    F -->|Yes| G[Create session]
    F -->|No| E
    G --> H[Redirect to dashboard]

Request: "Show the V0 interaction loop"
Response:
graph LR
    A[Ask] --> B[Retrieve]
    B --> C[Reason]
    C --> D[Produce Artifact]
    D --> E[Propose Memory]
    E --> A

Request: "API request flow"
Response:
flowchart LR
    A[Client] --> B[Send request]
    B --> C[Server receives]
    C --> D{Authenticated?}
    D -->|Yes| E[Process request]
    D -->|No| F[Return 401]
    E --> G[Return response]
"""

    user_prompt = f"Create a {req.syntax} diagram: {req.prompt}"
    full_prompt = f"{system_context}\n\n{user_prompt}"

    llm_response = await ask_llm(
        question=full_prompt,
        temperature=req.temperature,
        max_tokens=2000
    )

    raw_answer = llm_response["answer"]
    diagram_code, _ = _split_thinking(raw_answer)

    # Clean up any markdown code fences if the LLM added them
    if diagram_code.startswith("```"):
        lines = diagram_code.split("\n")
        # Remove first line (```mermaid or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        # Remove "mermaid" language identifier if present
        if lines and lines[0].strip().lower() in ["mermaid", "plantuml"]:
            lines = lines[1:]
        diagram_code = "\n".join(lines).strip()

    diagram_code = sanitize_mermaid(diagram_code)

    # Record diagram lineage in Neo4j (best-effort)
    diagram_id = f"diag-{uuid.uuid4().hex[:12]}"
    if req.source_session_id:
        try:
            await graph_queries.record_diagram_lineage(
                diagram_id=diagram_id,
                source_session_id=req.source_session_id,
                prompt=req.prompt,
                diagram_code=diagram_code,
                model=llm_response.get("model"),
            )
        except Exception:
            pass

    return {
        "syntax": req.syntax,
        "code": diagram_code,
        "diagram_id": diagram_id,
        "model": llm_response.get("model"),
        "processing_time": llm_response.get("processing_time")
    }


def sanitize_mermaid(code: str) -> str:
    """Fix common Mermaid syntax errors from LLM output.

    Handles:
    - Truncated output (unclosed brackets on last line)
    - Quotes inside [...] labels: [Print "hello"] → [Print hello]
    - Parentheses inside [...] labels: [Text (V0)] → [Text - V0]
    - Parentheses inside {...} labels: {Text (V0)?} → {Text - V0?}
    - Leading dash in rounded nodes: G(- End) → G(End)
    - Pipe chars inside labels: [|A| / n] → [❘A❘ / n] (Mermaid uses | for edge labels)
    - Duplicate node definitions on same line (DeepSeek quirk)
    - Semicolons as line separators
    - Redundant node ID suffixes in labels: A[Input - A] → A[Input]
    """
    # Remove truncated trailing lines (LLM hit max_tokens mid-node)
    lines = code.rstrip().split("\n")
    while lines:
        last = lines[-1]
        # Check for unclosed brackets: has [ without ], { without }, or ( without )
        if (last.count("[") > last.count("]")
                or last.count("{") > last.count("}")
                or last.count("(") > last.count(")")):
            lines.pop()
        else:
            break
    code = "\n".join(lines)

    # Split semicolon-separated statements into separate lines
    # e.g. "A --> B; B --> C" → "A --> B\n    B --> C"
    code = re.sub(r';\s*', '\n    ', code)

    # Fix leading dash/hyphen inside node parentheses: G(- End) → G(End)
    # Matches ID( - text) or ID(- text) patterns
    code = re.sub(r'(\w)\(\s*-\s+', lambda m: f'{m.group(1)}(', code)

    # Fix leading dash inside square brackets: A[- Text] → A[Text]
    code = re.sub(r'(\w)\[\s*-\s+', lambda m: f'{m.group(1)}[', code)

    # Strip quotes inside [...] node labels (quotes break Mermaid parser)
    def strip_quotes_in_brackets(match: re.Match) -> str:
        content = match.group(1)
        content = content.replace('"', '').replace("'", '')
        return f"[{content}]"
    code = re.sub(r'\[([^\]]*"[^\]]*)\]', strip_quotes_in_brackets, code)

    def fix_label(match: re.Match) -> str:
        opener = match.group(1)  # [ or {
        content = match.group(2)
        closer = match.group(3)  # ] or }

        # Replace parentheses inside labels with dashes
        content = content.replace("(", " - ").replace(")", "")
        # Collapse multiple spaces
        content = re.sub(r"\s{2,}", " ", content).strip()
        return f"{opener}{content}{closer}"

    # Fix [...] node labels containing parentheses
    code = re.sub(r'(\[)([^\]]*\([^\]]*?)(\])', fix_label, code)
    # Fix {...} decision labels containing parentheses
    code = re.sub(r'(\{)([^\}]*\([^\}]*?)(\})', fix_label, code)
    # Fix (...) rounded labels containing nested parentheses
    # e.g. (Text (V0)) — replace inner parens only
    code = re.sub(
        r'\(([^)]*)\(([^)]*)\)([^)]*)\)',
        lambda m: f"({m.group(1)}- {m.group(2)}{m.group(3)})",
        code,
    )

    # Replace pipe chars inside node labels — Mermaid treats | as edge label delimiters
    # e.g. AccuracyCalc[Calculate Accuracy = |A| / n] → AccuracyCalc[Calculate Accuracy = ❘A❘ / n]
    def fix_pipes_in_label(m: re.Match) -> str:
        nid = m.group(1)
        content = m.group(2)
        # Replace | inside the label with a visually similar Unicode char (U+2758)
        content = content.replace("|", "❘")
        return f"{nid}[{content}]"
    code = re.sub(r'([A-Za-z_]\w*)\[([^\]]*\|[^\]]*)\]', fix_pipes_in_label, code)

    # Same for {...} decision diamonds
    def fix_pipes_in_diamond(m: re.Match) -> str:
        nid = m.group(1)
        content = m.group(2)
        content = content.replace("|", "❘")
        return f"{nid}{{{content}}}"
    code = re.sub(r'([A-Za-z_]\w*)\{([^}]*\|[^}]*)\}', fix_pipes_in_diamond, code)

    # Remove redundant node ID suffixes in labels: A[Input - A] → A[Input]
    # DeepSeek often appends " - NodeID" to labels
    def strip_id_suffix(m: re.Match) -> str:
        nid = m.group(1)
        opener = m.group(2)
        label = m.group(3)
        closer = m.group(4)
        # Remove trailing " - NodeID" or " - NodeID?" pattern
        cleaned = re.sub(r'\s*-\s*' + re.escape(nid) + r'\??\s*$', '', label, flags=re.IGNORECASE)
        return f"{nid}{opener}{cleaned}{closer}"
    code = re.sub(r'([A-Za-z_]\w*)(\[)([^\]]+)(\])', strip_id_suffix, code)
    code = re.sub(r'([A-Za-z_]\w*)(\{)([^}]+)(\})', strip_id_suffix, code)

    # Fix bare (Text) nodes missing a node ID prefix
    # e.g. "    (Start) --> A[Step]" → "    _S([Start]) --> A[Step]"
    # Only matches at line-start position (after optional whitespace) or after --> arrows
    counter = [0]
    def add_node_id(m: re.Match) -> str:
        counter[0] += 1
        prefix = m.group(1)  # whitespace or arrow
        label = m.group(2)
        return f"{prefix}_N{counter[0]}([{label}])"
    # After whitespace at line start
    code = re.sub(r'(^[ \t]+)\(([A-Za-z][^)]*)\)', add_node_id, code, flags=re.MULTILINE)
    # After --> or ---
    code = re.sub(r'(-->[ \t]*)\(([A-Za-z][^)]*)\)', add_node_id, code)

    return code


# ============================================================================
# Graph Explorer Endpoints
# ============================================================================

@app.get("/graph/data", response_model=GraphData)
async def graph_data(limit: int = Query(default=100, ge=1, le=500)):
    """Return the full graph up to a limit."""
    return await graph_queries.get_full_graph(limit=limit)


@app.post("/graph/neighbors", response_model=GraphData)
async def graph_neighbors(req: NeighborRequest):
    """Expand neighborhood around a node."""
    return await graph_queries.get_neighbors(
        node_id=req.node_id, depth=req.depth, limit=req.limit
    )


@app.get("/graph/search", response_model=GraphData)
async def graph_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Search nodes by name/text."""
    return await graph_queries.search_nodes(query=q, limit=limit)


@app.post("/graph/concepts", response_model=GraphNode)
async def graph_create_concept(req: ConceptCreate):
    """Create or merge a Concept node."""
    return await graph_queries.create_concept(
        name=req.name, description=req.description, source=req.source
    )


@app.post("/graph/relationships", response_model=GraphEdge)
async def graph_create_relationship(req: RelationshipCreate):
    """Create a relationship between two nodes."""
    return await graph_queries.create_relationship(
        source_id=req.source_id,
        target_id=req.target_id,
        rel_type=req.rel_type,
        strength=req.strength,
    )


@app.get("/graph/stats")
async def graph_stats():
    """Return node and edge counts."""
    return await graph_queries.get_stats()


@app.post("/graph/sync-memories")
async def graph_sync_memories():
    """Bulk sync all approved memory cards to the knowledge graph."""
    approved = [c for c in MEMORY_CARDS.values() if c.approval == MemoryApproval.APPROVED]
    synced = 0
    errors = 0
    for card in approved:
        try:
            await graph_queries.sync_memory_card(card)
            synced += 1
        except Exception:
            errors += 1
    return {"synced": synced, "errors": errors, "total_approved": len(approved)}


@app.post("/graph/seed")
async def graph_seed():
    """Seed the knowledge graph with demo data."""
    return await graph_queries.seed_demo_data()


@app.get("/graph/session/{session_id}", response_model=GraphData)
async def graph_session_subgraph(session_id: str):
    """Return a focused subgraph around a chat session (1-2 hops)."""
    return await graph_queries.get_session_subgraph(session_id=session_id)


@app.get("/graph/scoped", response_model=GraphData)
async def graph_scoped(
    scope: str = Query(default="session", regex="^(session|question|artifact)$"),
    session_id: Optional[str] = Query(default=None),
    node_id: Optional[str] = Query(default=None),
    depth: int = Query(default=1, ge=1, le=3),
    view_mode: str = Query(default="provenance", regex="^(provenance|lineage|full)$"),
):
    """Return a scoped, predictable subgraph.

    scope: session | question | artifact
    view_mode: provenance | lineage | full
    depth: 1-3 hops from the anchor node
    """
    return await graph_queries.get_scoped_graph(
        scope=scope,
        session_id=session_id,
        node_id=node_id,
        depth=depth,
        view_mode=view_mode,
    )


@app.get("/graph/lineage/{node_id:path}", response_model=GraphData)
async def graph_lineage(node_id: str):
    """Get artifact lineage chain for a node."""
    return await graph_queries.get_artifact_lineage(node_id=node_id)


@app.post("/graph/from-diagram", response_model=GraphData)
async def graph_from_diagram(req: DiagramToGraphRequest):
    """Import a Mermaid diagram into the knowledge graph as Concept nodes + edges."""
    return await graph_queries.import_diagram_to_graph(
        diagram_code=req.diagram_code,
        source=req.source or "diagram_import",
    )


@app.post("/graph/to-diagram")
async def graph_to_diagram(req: GraphToDiagramRequest):
    """Export graph nodes/edges as Mermaid flowchart syntax."""
    ids = req.node_ids if req.node_ids is not None else None
    code = await graph_queries.export_graph_to_mermaid(
        node_ids=ids,
        depth=req.depth,
        layout=req.layout,
    )
    return {"code": code}


# ============================================================================
# Session Persistence Endpoints
# ============================================================================

class SaveSessionRequest(BaseModel):
    session_id: str
    title: str
    messages: List[dict]


@app.post("/sessions/save")
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


@app.get("/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Load a chat session by ID."""
    session = await load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump(mode="json")


@app.get("/sessions")
async def get_sessions_list(limit: int = Query(default=50, ge=1, le=200)):
    """List all sessions (summaries without full messages)."""
    return await list_sessions(limit=limit)


@app.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.get("/sessions/{session_id}/export")
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


# ============================================================================
# Mind File Endpoints
# ============================================================================

@app.get("/mindfile")
async def list_mindfile(
    category: Optional[str] = Query(default=None),
):
    """List all Mind File entries, optionally filtered by category."""
    cat = MindFileEntryCategory(category) if category else None
    entries = await load_mindfile_entries(category=cat)
    return [e.model_dump(mode="json") for e in entries]


@app.get("/mindfile/stats")
async def mindfile_stats():
    """Return Mind File statistics."""
    entries = await load_mindfile_entries()
    by_category: dict[str, int] = {}
    for e in entries:
        by_category[e.category.value] = by_category.get(e.category.value, 0) + 1
    first_date = min((e.created_at for e in entries), default=None)
    latest_date = max((e.created_at for e in entries), default=None)
    return {
        "total": len(entries),
        "by_category": by_category,
        "first_entry_date": first_date.isoformat() if first_date else None,
        "latest_entry_date": latest_date.isoformat() if latest_date else None,
    }


@app.get("/mindfile/export")
async def export_mindfile():
    """Export the full Mind File as structured markdown."""
    entries = await load_mindfile_entries()
    lines: list[str] = []
    lines.append("# Mind File")
    lines.append("")
    lines.append(f"**Total entries:** {len(entries)}")
    lines.append("")

    # Group by category
    grouped: dict[str, list[MindFileEntry]] = {}
    for e in entries:
        grouped.setdefault(e.category.value, []).append(e)

    category_labels = {
        "principles_values": "Principles & Values",
        "cognitive_framing": "Cognitive Framing",
        "decision_heuristics": "Decision Heuristics",
        "preferences": "Preferences",
        "vocabulary": "Vocabulary",
    }

    for cat_value, label in category_labels.items():
        cat_entries = grouped.get(cat_value, [])
        if not cat_entries:
            continue
        lines.append(f"## {label}")
        lines.append("")
        for e in cat_entries:
            lines.append(f"- {e.text}")
            if e.note:
                lines.append(f"  - *Note: {e.note}*")
            lines.append(f"  - Added: {e.created_at.strftime('%Y-%m-%d')}")
            lines.append("")
        lines.append("")

    from datetime import datetime
    lines.append(f"*Exported from Noesis on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")

    markdown = "\n".join(lines)
    return {
        "markdown": markdown,
        "filename": "mind_file.md",
        "entry_count": len(entries),
    }


@app.post("/mindfile/backfill")
async def backfill_mindfile():
    """One-time backfill: create Mind File entries for all approved cards that don't have one."""
    approved = [c for c in MEMORY_CARDS.values() if c.approval == MemoryApproval.APPROVED]
    created = 0
    skipped = 0
    for card in approved:
        if await entry_exists_for_card(card.id):
            skipped += 1
            continue
        entry = MindFileEntry(
            category=MindFileEntryCategory(card.category.value),
            text=card.text,
            source_memory_card_id=card.id,
        )
        await save_mindfile_entry(entry)
        try:
            await graph_queries.sync_mindfile_entry(entry)
        except Exception:
            pass
        created += 1
    return {"created": created, "skipped": skipped, "total_approved": len(approved)}


@app.post("/mindfile/sync-graph")
async def sync_mindfile_to_graph():
    """Sync all existing Mind File entries to Neo4j graph (MindFileEntry nodes + edges)."""
    entries = await load_mindfile_entries()
    synced = 0
    errors = 0
    for entry in entries:
        try:
            await graph_queries.sync_mindfile_entry(entry)
            synced += 1
        except Exception:
            errors += 1
    return {"synced": synced, "errors": errors, "total": len(entries)}


# ============================================================================
# Room 1: Timeline, Patterns, Cognitive Profile
# (These MUST be declared before the /mindfile/{entry_id} wildcard route)
# ============================================================================

@app.get("/mindfile/timeline/{concept}")
async def concept_timeline(concept: str):
    """Return a chronological timeline of all artifacts related to a concept."""
    events = await graph_queries.get_concept_timeline(concept)
    return events


@app.get("/mindfile/patterns")
async def mindfile_patterns():
    """Return concept co-occurrence patterns and top concepts."""
    top_concepts = await graph_queries.get_top_concepts(limit=20)
    co_occurrences = await graph_queries.get_concept_cooccurrences(limit=20)
    category_trend = await graph_queries.get_category_trend()
    return {
        "top_concepts": top_concepts,
        "co_occurrences": co_occurrences,
        "category_trend": category_trend,
    }


@app.get("/mindfile/cognitive-profile")
async def cognitive_profile():
    """Return an aggregate cognitive style summary."""
    entries = await load_mindfile_entries()
    category_breakdown: dict[str, int] = {}
    for e in entries:
        category_breakdown[e.category.value] = category_breakdown.get(e.category.value, 0) + 1

    top_concepts = await graph_queries.get_top_concepts(limit=10)
    co_occurrences = await graph_queries.get_concept_cooccurrences(limit=10)
    category_trend = await graph_queries.get_category_trend()

    first_date = min((e.created_at for e in entries), default=None)
    latest_date = max((e.created_at for e in entries), default=None)

    return {
        "total_entries": len(entries),
        "category_breakdown": category_breakdown,
        "top_concepts": top_concepts,
        "co_occurrences": co_occurrences,
        "category_trend": category_trend,
        "first_entry_date": first_date.isoformat() if first_date else None,
        "latest_entry_date": latest_date.isoformat() if latest_date else None,
    }


# Wildcard entry routes — MUST come after all /mindfile/... specific routes
class UpdateNoteRequest(BaseModel):
    note: str


@app.get("/mindfile/{entry_id}")
async def get_mindfile_entry(entry_id: str):
    """Get a single Mind File entry."""
    entry = await load_mindfile_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry.model_dump(mode="json")


@app.put("/mindfile/{entry_id}/note")
async def update_mindfile_entry_note(entry_id: str, req: UpdateNoteRequest):
    """Update the user annotation on a Mind File entry."""
    entry = await update_mindfile_note(entry_id, req.note)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry.model_dump(mode="json")


@app.delete("/mindfile/{entry_id}")
async def remove_mindfile_entry(entry_id: str):
    """Remove an entry from the Mind File. The source memory card stays approved."""
    deleted = await delete_mindfile_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "deleted", "entry_id": entry_id}