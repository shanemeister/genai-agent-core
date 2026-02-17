"""Chat, streaming, validation, and diagram generation endpoints."""

from __future__ import annotations

import json as json_mod
import logging
import re
import time
import uuid
from typing import List, Optional

import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from core.artifacts.memory_card import (
    MemoryApproval,
    MemoryCard,
    MemoryCategory,
    MemoryProvenance,
    MemoryScope,
)
from core.artifacts.storage_memory_pg import upsert_card
from core.config import settings
from core.graph import queries as graph_queries
from core.validation.models import ValidationResult
from core.api.shared import MEMORY_CARDS, ask_llm

log = logging.getLogger("noesis.chat")

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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


class ValidateRequest(BaseModel):
    response_text: str
    user_question: str
    max_claims: int = 6


class DiagramRequest(BaseModel):
    prompt: str
    syntax: str = "mermaid"
    temperature: float = 0.3
    source_session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Compute a grounding score for a chat response."""
    # --- Retrieval quality ---
    raw_scores = [d["score"] for d in context_docs if d.get("score")]
    if raw_scores:
        max_s = max(raw_scores)
        if max_s > 1.5:
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
    except Exception as e:
        log.warning("Graph coverage check failed: %s", e)
        coverage = 0.0

    # --- Source diversity ---
    unique_sources = set()
    for d in context_docs:
        doc_id = d.get("doc_id", "")
        if doc_id.startswith("memory:"):
            unique_sources.add(doc_id)
        else:
            unique_sources.add("seed")
    n = len(unique_sources)
    diversity = min(n / 4.0, 1.0) if n > 0 else 0.0

    # --- Reasoning ---
    reasoning_score = 1.0 if has_reasoning else 0.0

    # --- Composite ---
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
            except Exception as e:
                log.warning("Failed to link memory %s to session: %s", card.id, e)
    except Exception as e:
        log.warning("Memory proposal failed for session %s: %s", session_id, e)
    return proposed_cards


def sanitize_mermaid(code: str) -> str:
    """Fix common Mermaid syntax errors from LLM output."""
    # Remove truncated trailing lines (LLM hit max_tokens mid-node)
    lines = code.rstrip().split("\n")
    while lines:
        last = lines[-1]
        if (last.count("[") > last.count("]")
                or last.count("{") > last.count("}")
                or last.count("(") > last.count(")")):
            lines.pop()
        else:
            break
    code = "\n".join(lines)

    # Split semicolon-separated statements into separate lines
    code = re.sub(r';\s*', '\n    ', code)

    # Fix leading dash/hyphen inside node parentheses: G(- End) → G(End)
    code = re.sub(r'(\w)\(\s*-\s+', lambda m: f'{m.group(1)}(', code)

    # Fix leading dash inside square brackets: A[- Text] → A[Text]
    code = re.sub(r'(\w)\[\s*-\s+', lambda m: f'{m.group(1)}[', code)

    # Strip quotes inside [...] node labels
    def strip_quotes_in_brackets(match: re.Match) -> str:
        content = match.group(1)
        content = content.replace('"', '').replace("'", '')
        return f"[{content}]"
    code = re.sub(r'\[([^\]]*"[^\]]*)\]', strip_quotes_in_brackets, code)

    def fix_label(match: re.Match) -> str:
        opener = match.group(1)
        content = match.group(2)
        closer = match.group(3)
        content = content.replace("(", " - ").replace(")", "")
        content = re.sub(r"\s{2,}", " ", content).strip()
        return f"{opener}{content}{closer}"

    # Fix [...] node labels containing parentheses
    code = re.sub(r'(\[)([^\]]*\([^\]]*?)(\])', fix_label, code)
    # Fix {...} decision labels containing parentheses
    code = re.sub(r'(\{)([^\}]*\([^\}]*?)(\})', fix_label, code)
    # Fix (...) rounded labels containing nested parentheses
    code = re.sub(
        r'\(([^)]*)\(([^)]*)\)([^)]*)\)',
        lambda m: f"({m.group(1)}- {m.group(2)}{m.group(3)})",
        code,
    )

    # Replace pipe chars inside node labels
    def fix_pipes_in_label(m: re.Match) -> str:
        nid = m.group(1)
        content = m.group(2)
        content = content.replace("|", "\u2758")
        return f"{nid}[{content}]"
    code = re.sub(r'([A-Za-z_]\w*)\[([^\]]*\|[^\]]*)\]', fix_pipes_in_label, code)

    def fix_pipes_in_diamond(m: re.Match) -> str:
        nid = m.group(1)
        content = m.group(2)
        content = content.replace("|", "\u2758")
        return f"{nid}{{{content}}}"
    code = re.sub(r'([A-Za-z_]\w*)\{([^}]*\|[^}]*)\}', fix_pipes_in_diamond, code)

    # Remove redundant node ID suffixes in labels
    def strip_id_suffix(m: re.Match) -> str:
        nid = m.group(1)
        opener = m.group(2)
        label = m.group(3)
        closer = m.group(4)
        cleaned = re.sub(r'\s*-\s*' + re.escape(nid) + r'\??\s*$', '', label, flags=re.IGNORECASE)
        return f"{nid}{opener}{cleaned}{closer}"
    code = re.sub(r'([A-Za-z_]\w*)(\[)([^\]]+)(\])', strip_id_suffix, code)
    code = re.sub(r'([A-Za-z_]\w*)(\{)([^}]+)(\})', strip_id_suffix, code)

    # Fix bare (Text) nodes missing a node ID prefix
    counter = [0]
    def add_node_id(m: re.Match) -> str:
        counter[0] += 1
        prefix = m.group(1)
        label = m.group(2)
        return f"{prefix}_N{counter[0]}([{label}])"
    code = re.sub(r'(^[ \t]+)\(([A-Za-z][^)]*)\)', add_node_id, code, flags=re.MULTILINE)
    code = re.sub(r'(-->[ \t]*)\(([A-Za-z][^)]*)\)', add_node_id, code)

    return code


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint - implements the core Ask → Retrieve → Reason → Respond loop."""
    # 1. Retrieve context from RAG
    from core.rag.retriever import retrieve_context, index_memory_cards
    await index_memory_cards(
        [c for c in MEMORY_CARDS.values() if c.approval == MemoryApproval.APPROVED]
    )
    context_docs = await retrieve_context(req.message, k=5)
    context_text = "\n".join([f"- {doc['text']}" for doc in context_docs])

    # 2. Build prompt with context
    prompt = _build_chat_prompt(req.message, context_text)

    # 3. Call LLM
    llm_response = await ask_llm(
        question=prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
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
    except Exception as e:
        log.warning("Chat lineage recording failed: %s", e)

    # 6. Auto-propose memory cards (best-effort)
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
    except Exception as e:
        log.warning("Grounding score computation failed: %s", e)

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


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming chat — tokens appear in real-time."""
    # 1. RAG retrieval
    from core.rag.retriever import retrieve_context, index_memory_cards
    await index_memory_cards(
        [c for c in MEMORY_CARDS.values() if c.approval == MemoryApproval.APPROVED]
    )
    context_docs = await retrieve_context(req.message, k=5)
    context_text = "\n".join([f"- {doc['text']}" for doc in context_docs])
    prompt = _build_chat_prompt(req.message, context_text)

    retrieved = [
        {"doc_id": d["doc_id"], "text": d["text"], "score": round(d["score"], 3)}
        for d in context_docs
    ]

    async def generate():
        start = time.time()

        yield f"event: status\ndata: {json_mod.dumps({'phase': 'retrieving'})}\n\n"

        full_text = ""
        answer_started = False
        sent_thinking_status = False

        vllm_payload = {
            "model": settings.vllm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                async with client.stream(
                    "POST",
                    f"{settings.vllm_base_url}/v1/chat/completions",
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
                            yield f"event: token\ndata: {json_mod.dumps({'token': token})}\n\n"
                            continue

                        if "</think>" in full_text:
                            answer_started = True
                            yield f"event: status\ndata: {json_mod.dumps({'phase': 'answering'})}\n\n"
                            parts = full_text.split("</think>", 1)
                            after = parts[1].lstrip() if len(parts) > 1 else ""
                            if after:
                                yield f"event: token\ndata: {json_mod.dumps({'token': after})}\n\n"
                            continue

                        if not sent_thinking_status:
                            sent_thinking_status = True
                            yield f"event: status\ndata: {json_mod.dumps({'phase': 'thinking'})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json_mod.dumps({'detail': str(e)})}\n\n"
            return

        # Post-processing
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
        except Exception as e:
            log.warning("Stream chat lineage recording failed: %s", e)

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
        except Exception as e:
            log.warning("Stream grounding score failed: %s", e)

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


@router.post("/validate", response_model=ValidationResult)
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


@router.post("/diagram/generate")
async def generate_diagram(req: DiagramRequest):
    """Generate Mermaid diagram syntax from a natural language prompt."""
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
        max_tokens=2000,
    )

    raw_answer = llm_response["answer"]
    diagram_code, _ = _split_thinking(raw_answer)

    # Clean up any markdown code fences if the LLM added them
    if diagram_code.startswith("```"):
        lines = diagram_code.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
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
        except Exception as e:
            log.warning("Diagram lineage recording failed: %s", e)

    return {
        "syntax": req.syntax,
        "code": diagram_code,
        "diagram_id": diagram_id,
        "model": llm_response.get("model"),
        "processing_time": llm_response.get("processing_time"),
    }
