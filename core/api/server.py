from __future__ import annotations
import json as json_mod
import os
import re
import time
import uuid
from core.artifacts.storage_sqlite import load_all_cards, upsert_card
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import httpx

from core.artifacts.memory_card import (
    MemoryApproval,
    MemoryCard,
    MemoryCategory,
    MemoryProvenance,
    MemoryScope,
)
from core.graph.models import (
    ConceptCreate,
    GraphData,
    GraphEdge,
    GraphNode,
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

class ChatResponse(BaseModel):
    session_id: str
    response: str
    reasoning: Optional[str] = None
    model: Optional[str] = None
    processing_time: Optional[float] = None
    retrieved_context: List[RetrievedDoc] = []
    proposed_memories: List[MemoryCard] = []
    proposed_diagram: Optional[str] = None


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

    return ChatResponse(
        session_id=session_id,
        response=visible_answer,
        reasoning=reasoning,
        model=llm_response.get("model"),
        processing_time=llm_response.get("processing_time"),
        retrieved_context=retrieved,
        proposed_memories=proposed_cards,
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

        # Send final done event with all metadata
        done_payload = {
            "session_id": session_id,
            "reasoning": reasoning,
            "model": "DeepSeek-R1-70B",
            "processing_time": elapsed,
            "retrieved_context": retrieved,
            "proposed_memories": proposed_dicts,
        }
        yield f"event: done\ndata: {json_mod.dumps(done_payload)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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


@app.get("/graph/lineage/{node_id:path}", response_model=GraphData)
async def graph_lineage(node_id: str):
    """Get artifact lineage chain for a node."""
    return await graph_queries.get_artifact_lineage(node_id=node_id)