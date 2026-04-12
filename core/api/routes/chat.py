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

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []  # Prior turns for conversation context
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

    Thinking models (DeepSeek-R1, Qwen3.5) may emit chain-of-thought
    in <think> tags when served via vLLM. Ollama separates thinking
    into a different response field, so content arrives clean.
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


def _build_chat_prompt(
    user_message: str,
    context_text: str,
    history: list[dict] | None = None,
) -> str:
    """Build the full system + context + history + user prompt.

    If history is provided, it is rendered as a "Conversation so far" block
    before the current user question, giving the LLM context about what
    has been discussed in the session.
    """
    history_block = ""
    if history:
        # Cap each message to prevent prompt bloat (last 8KB of any single turn)
        turns = []
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = (msg.get("content") or "")[:8000]
            turns.append(f"{role}: {content}")
        history_block = (
            "\nConversation so far (oldest first):\n"
            + "\n\n".join(turns)
            + "\n"
        )

    return f"""You are Noesis, a private AI assistant for knowledge organization and reasoning. You run locally and prioritize accuracy over helpfulness.

CRITICAL RULES:
- Only state facts you are confident about. If you are uncertain or the topic is outside your training data, say "I'm not confident about this" and explain what you do know vs. what you're unsure about.
- Never fabricate definitions, dates, or technical details. Getting something wrong is worse than admitting uncertainty.
- Your training data has a cutoff date. If asked about recent events, products, or updates, acknowledge that your information may be outdated.
- Use the retrieved context below when relevant, but don't force it into answers where it doesn't apply.
- If a diagram or graph would help explain your response, mention it.
- If the user refers to something from earlier in the conversation (e.g., "yes", "the previous point", "that example"), use the Conversation so far section to understand what they mean — then call the appropriate tools again to get current data, rather than re-using information from memory.
- IMPORTANT: When asked about a specific Noesis CDI rule (by ID or condition) OR asked to show an example query/template, ALWAYS call the rules_get_detail or rules_find_by_condition tool to fetch the current rule data, even if the conversation history already mentions that rule. Never fabricate query template text — use what the tool returns.

Retrieved context from the knowledge base:
{context_text}
{history_block}
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
    history_dicts = [m.model_dump() for m in req.history] if req.history else None
    prompt = _build_chat_prompt(req.message, context_text, history=history_dicts)

    # 3. Call LLM with tool-use support (same framework as streaming endpoint)
    from core.tools.snomed_tools import get_all_tool_definitions, execute_tool
    import time as time_mod

    start_time = time_mod.time()
    tools = get_all_tool_definitions()
    messages = [{"role": "user", "content": prompt}]
    tool_calls_log = []
    effective_max = max(req.max_tokens * 4, 8192)

    MAX_TOOL_ROUNDS = 3
    finish_reason = ""
    msg = {}

    # First call with tools
    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
        resp = await client.post(
            f"{settings.llm_base_url}/v1/chat/completions",
            json={
                "model": settings.llm_model_name,
                "messages": messages,
                "tools": tools,
                "temperature": req.temperature,
                "max_tokens": effective_max,
                "stream": False,
            },
            timeout=settings.llm_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        msg = choice.get("message", {})

    # Tool-call loop
    for tool_round in range(MAX_TOOL_ROUNDS):
        if finish_reason != "tool_calls" or not msg.get("tool_calls"):
            break
        messages.append(msg)

        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            try:
                tool_args = json_mod.loads(fn.get("arguments", "{}"))
            except (json_mod.JSONDecodeError, TypeError):
                tool_args = {}

            log.info("Tool call (round %d): %s(%s)", tool_round + 1, tool_name, tool_args)
            result_json = await execute_tool(tool_name, tool_args)

            tool_calls_log.append({
                "tool": tool_name, "args": tool_args,
                "result_length": len(result_json), "round": tool_round + 1,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result_json,
            })

        # Next LLM call with tool results
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            resp = await client.post(
                f"{settings.llm_base_url}/v1/chat/completions",
                json={
                    "model": settings.llm_model_name,
                    "messages": messages,
                    "tools": tools,
                    "temperature": req.temperature,
                    "max_tokens": effective_max,
                    "stream": False,
                },
                timeout=settings.llm_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason", "")
            msg = choice.get("message", {})

    # If we exhausted tool rounds and the LLM still wants tools,
    # make one final call WITHOUT tools to force a text answer
    if finish_reason == "tool_calls" and msg.get("tool_calls"):
        messages.append(msg)
        # Add a synthetic tool response that says "no more tools available"
        for tc in msg["tool_calls"]:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": '{"note": "Tool call limit reached. Please answer with the information you have gathered so far."}',
            })
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            resp = await client.post(
                f"{settings.llm_base_url}/v1/chat/completions",
                json={
                    "model": settings.llm_model_name,
                    "messages": messages,
                    "temperature": req.temperature,
                    "max_tokens": effective_max,
                    "stream": False,
                    # NOTE: no "tools" key — forces text generation
                },
                timeout=settings.llm_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0].get("message", {})

    # 3b. Separate <think> reasoning from visible answer
    raw_answer = msg.get("content", "") or ""
    visible_answer, reasoning = _split_thinking(raw_answer)
    processing_time = round(time_mod.time() - start_time, 2)

    # Build a compatible llm_response dict for the rest of the pipeline
    llm_response = {
        "answer": raw_answer,
        "model": settings.llm_model_display,
        "processing_time": processing_time,
    }

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

    # 7. Grounding score — tool-aware
    grounding = None
    try:
        if tool_calls_log:
            total_result_bytes = sum(tc.get("result_length", 0) for tc in tool_calls_log)
            tool_count = len(tool_calls_log)
            max_round = max(tc.get("round", 1) for tc in tool_calls_log)
            exhausted = (max_round >= MAX_TOOL_ROUNDS)

            if total_result_bytes > 500 and not exhausted:
                score, label = 0.95, "High"
                detail = f"Tool-grounded: {tool_count} tool calls returned {total_result_bytes:,} bytes of structured SNOMED data"
            elif total_result_bytes > 100:
                score, label = 0.70, "Medium"
                detail = f"Partially tool-grounded: {tool_count} calls, {total_result_bytes:,} bytes"
            else:
                score, label = 0.35, "Low"
                detail = f"Tools returned minimal data ({total_result_bytes} bytes)"

            grounding = GroundingScore(
                overall=score, retrieval=0.0, coverage=0.0,
                diversity=0.0, reasoning=1.0 if reasoning else 0.0,
                label=label, detail=detail,
            )
        else:
            grounding = await _compute_grounding(
                query=req.message,
                context_docs=context_docs,
                has_reasoning=reasoning is not None,
            )
    except Exception as e:
        log.warning("Grounding score computation failed: %s", e)

    # 8. Observability: log this LLM call (best-effort)
    try:
        from core.db.llm_logger import log_llm_call
        pt = llm_response.get("processing_time")
        await log_llm_call(
            caller="chat",
            session_id=session_id,
            model=llm_response.get("model"),
            prompt=prompt,
            response=visible_answer,
            reasoning=reasoning,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            duration_ms=int(pt * 1000) if pt else None,
            grounding_score=grounding.overall if grounding else None,
            tool_calls=tool_calls_log if tool_calls_log else None,
        )
    except Exception as e:
        log.warning("LLM call logging failed: %s", e)

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
    history_dicts = [m.model_dump() for m in req.history] if req.history else None
    prompt = _build_chat_prompt(req.message, context_text, history=history_dicts)

    retrieved = [
        {"doc_id": d["doc_id"], "text": d["text"], "score": round(d["score"], 3)}
        for d in context_docs
    ]

    async def generate():
        start = time.time()

        yield f"event: status\ndata: {json_mod.dumps({'phase': 'retrieving'})}\n\n"

        # ── Tool-use: first LLM call (non-streaming) to check if tools are needed ──
        from core.tools.snomed_tools import get_all_tool_definitions, execute_tool

        effective_max = max(req.max_tokens * 4, 8192)
        tools = get_all_tool_definitions()
        messages = [{"role": "user", "content": prompt}]
        tool_calls_log = []  # For observability

        # Phase 1: Ask the LLM with tools available (non-streaming so we
        # can cleanly detect tool_calls in the response)
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                tool_check = await client.post(
                    f"{settings.llm_base_url}/v1/chat/completions",
                    json={
                        "model": settings.llm_model_name,
                        "messages": messages,
                        "tools": tools,
                        "temperature": req.temperature,
                        "max_tokens": effective_max,
                        "stream": False,
                    },
                    timeout=settings.llm_timeout,
                )
                tool_check.raise_for_status()
                tool_response = tool_check.json()
                choice = tool_response["choices"][0]
                finish_reason = choice.get("finish_reason", "")
                msg = choice.get("message", {})
        except Exception as e:
            yield f"event: error\ndata: {json_mod.dumps({'detail': f'LLM tool-check failed: {e}'})}\n\n"
            return

        # Phase 2: Tool-call loop — allow up to 3 rounds of tool calls.
        # The LLM may call search first, then get_descendants with the SCTID
        # it found. Each round: execute tools, feed results back, ask the LLM
        # again (non-streaming) until it either produces content or we hit the limit.
        MAX_TOOL_ROUNDS = 3
        _skip_streaming = False

        for tool_round in range(MAX_TOOL_ROUNDS):
            if finish_reason != "tool_calls" or not msg.get("tool_calls"):
                break  # LLM is done with tools

            yield f"event: status\ndata: {json_mod.dumps({'phase': 'thinking', 'detail': f'Calling tools (round {tool_round + 1})...'})}\n\n"

            # Add the assistant's tool-call message to the conversation
            messages.append(msg)

            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    tool_args = json_mod.loads(fn.get("arguments", "{}"))
                except (json_mod.JSONDecodeError, TypeError):
                    tool_args = {}

                log.info("Tool call (round %d): %s(%s)", tool_round + 1, tool_name, tool_args)

                result_json = await execute_tool(tool_name, tool_args)

                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_length": len(result_json),
                    "round": tool_round + 1,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result_json,
                })

            # Ask the LLM again with the tool results — it may request more
            # tools or produce the final answer
            try:
                async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                    next_resp = await client.post(
                        f"{settings.llm_base_url}/v1/chat/completions",
                        json={
                            "model": settings.llm_model_name,
                            "messages": messages,
                            "tools": tools,
                            "temperature": req.temperature,
                            "max_tokens": effective_max,
                            "stream": False,
                        },
                        timeout=settings.llm_timeout,
                    )
                    next_resp.raise_for_status()
                    next_data = next_resp.json()
                    choice = next_data["choices"][0]
                    finish_reason = choice.get("finish_reason", "")
                    msg = choice.get("message", {})
            except Exception as e:
                yield f"event: error\ndata: {json_mod.dumps({'detail': f'Tool follow-up failed: {e}'})}\n\n"
                return

        # If we exhausted tool rounds and the LLM still wants more,
        # force a final answer without tools
        if finish_reason == "tool_calls" and msg.get("tool_calls"):
            messages.append(msg)
            for tc in msg["tool_calls"]:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": '{"note": "Tool call limit reached. Please answer with the information you have gathered so far."}',
                })
            try:
                async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                    force_resp = await client.post(
                        f"{settings.llm_base_url}/v1/chat/completions",
                        json={
                            "model": settings.llm_model_name,
                            "messages": messages,
                            "temperature": req.temperature,
                            "max_tokens": effective_max,
                            "stream": False,
                        },
                        timeout=settings.llm_timeout,
                    )
                    force_resp.raise_for_status()
                    force_data = force_resp.json()
                    msg = force_data["choices"][0].get("message", {})
                    finish_reason = force_data["choices"][0].get("finish_reason", "")
            except Exception as e:
                yield f"event: error\ndata: {json_mod.dumps({'detail': f'Final answer generation failed: {e}'})}\n\n"
                return

        # After the tool loop, check what we have
        if msg.get("content"):
            raw_answer = msg["content"]
            visible_answer_direct, reasoning_direct = _split_thinking(raw_answer)

            if reasoning_direct:
                yield f"event: status\ndata: {json_mod.dumps({'phase': 'thinking'})}\n\n"

            yield f"event: status\ndata: {json_mod.dumps({'phase': 'answering'})}\n\n"
            yield f"event: token\ndata: {json_mod.dumps({'token': visible_answer_direct})}\n\n"

            elapsed = round(time.time() - start, 2)
            reasoning = reasoning_direct
            visible_answer = visible_answer_direct
            _skip_streaming = True

        else:
            yield f"event: token\ndata: {json_mod.dumps({'token': '(No response)'})}\n\n"
            elapsed = round(time.time() - start, 2)
            reasoning = None
            visible_answer = "(No response)"
            _skip_streaming = True

        # Phase 3: Stream the final answer (either after tool results, or
        # as the original response if no tools were called)
        full_text = ""
        reasoning_text = ""
        answer_started = False
        sent_thinking_status = False

        if not _skip_streaming:
            # We have tool results in `messages` — now stream the final answer
            yield f"event: status\ndata: {json_mod.dumps({'phase': 'answering'})}\n\n"

            try:
                async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{settings.llm_base_url}/v1/chat/completions",
                        json={
                            "model": settings.llm_model_name,
                            "messages": messages,
                            "temperature": req.temperature,
                            "max_tokens": effective_max,
                            "stream": True,
                        },
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

                            reasoning_token = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                            if reasoning_token:
                                reasoning_text += reasoning_token
                                if not sent_thinking_status:
                                    sent_thinking_status = True
                                    yield f"event: status\ndata: {json_mod.dumps({'phase': 'thinking'})}\n\n"
                                continue

                            token = delta.get("content", "")
                            if not token:
                                continue

                            if sent_thinking_status and not answer_started:
                                answer_started = True
                                yield f"event: status\ndata: {json_mod.dumps({'phase': 'answering'})}\n\n"

                            full_text += token

                            if answer_started:
                                yield f"event: token\ndata: {json_mod.dumps({'token': token})}\n\n"
                                continue

                            if "</think>" in full_text:
                                answer_started = True
                                yield f"event: status\ndata: {json_mod.dumps({'phase': 'answering'})}\n\n"
                                parts = full_text.split("</think>", 1)
                                reasoning_text = parts[0].replace("<think>", "").strip()
                                after = parts[1].lstrip() if len(parts) > 1 else ""
                                full_text = after
                                if after:
                                    yield f"event: token\ndata: {json_mod.dumps({'token': after})}\n\n"
                                continue

                            if not sent_thinking_status:
                                sent_thinking_status = True
                                yield f"event: status\ndata: {json_mod.dumps({'phase': 'thinking'})}\n\n"

            except Exception as e:
                yield f"event: error\ndata: {json_mod.dumps({'detail': str(e)})}\n\n"
                return

            elapsed = round(time.time() - start, 2)
            if reasoning_text:
                visible_answer = full_text.strip()
                reasoning = reasoning_text.strip()
            else:
                visible_answer, reasoning = _split_thinking(full_text)

        # Post-processing — visible_answer and reasoning are set by
        # whichever path executed above (streaming or non-streaming)
        session_id = f"chat-{uuid.uuid4().hex[:12]}"

        # Neo4j lineage (best-effort)
        try:
            await graph_queries.record_chat_session(
                session_id=session_id,
                user_message=req.message,
                assistant_response=visible_answer,
                model=settings.llm_model_display,
                processing_time=elapsed,
                retrieved_doc_ids=[d["doc_id"] for d in context_docs],
            )
        except Exception as e:
            log.warning("Stream chat lineage recording failed: %s", e)

        # Memory proposals (best-effort)
        proposed_cards = await _propose_memories_for_session(
            req.message, visible_answer, session_id, settings.llm_model_display
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
        # When tools were used, compute grounding based on tool success
        # rather than RAG retrieval quality (which measured the wrong thing).
        grounding_dict = None
        try:
            if tool_calls_log:
                # Tools were called — assess grounding from tool results
                total_result_bytes = sum(tc.get("result_length", 0) for tc in tool_calls_log)
                tool_count = len(tool_calls_log)
                max_round = max(tc.get("round", 1) for tc in tool_calls_log)
                exhausted_rounds = (max_round >= MAX_TOOL_ROUNDS)

                if total_result_bytes > 500 and not exhausted_rounds:
                    # Tools returned substantial data and didn't exhaust rounds
                    tool_score = 0.95
                    label = "High"
                    detail = f"Tool-grounded: {tool_count} tool calls returned {total_result_bytes:,} bytes of structured SNOMED data"
                elif total_result_bytes > 100:
                    # Tools returned some data but either exhausted rounds or data was thin
                    tool_score = 0.70
                    label = "Medium"
                    detail = f"Partially tool-grounded: {tool_count} tool calls, {total_result_bytes:,} bytes. {'Exhausted tool rounds — answer includes general knowledge.' if exhausted_rounds else ''}"
                else:
                    # Tools were called but returned little useful data
                    tool_score = 0.35
                    label = "Low"
                    detail = f"Tools returned minimal data ({total_result_bytes} bytes). Answer relies primarily on general knowledge."

                grounding_dict = {
                    "overall": tool_score,
                    "retrieval": 0.0,
                    "coverage": 0.0,
                    "diversity": 0.0,
                    "reasoning": 1.0 if reasoning else 0.0,
                    "label": label,
                    "detail": detail,
                }
            else:
                # No tools used — compute RAG grounding as before
                grounding = await _compute_grounding(
                    query=req.message,
                    context_docs=context_docs,
                    has_reasoning=reasoning is not None,
                )
                grounding_dict = grounding.model_dump()
        except Exception as e:
            log.warning("Stream grounding score failed: %s", e)

        # ── Observability: log this LLM call ──────────────────
        # Best-effort — failures here never break the chat pipeline.
        llm_call_id = None
        try:
            from core.db.llm_logger import log_llm_call
            llm_call_id = await log_llm_call(
                caller="chat_stream",
                session_id=session_id,
                model=settings.llm_model_display,
                prompt=prompt,
                response=visible_answer,
                reasoning=reasoning,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                duration_ms=int(elapsed * 1000),
                grounding_score=(
                    grounding_dict.get("overall") if grounding_dict else None
                ),
                tool_calls=tool_calls_log if tool_calls_log else None,
            )
        except Exception as e:
            log.warning("LLM call logging failed: %s", e)

        done_payload = {
            "session_id": session_id,
            "llm_call_id": str(llm_call_id) if llm_call_id else None,
            "reasoning": reasoning,
            "model": settings.llm_model_display,
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


# ---------------------------------------------------------------------------
# Feedback endpoint — user thumbs up/down on LLM responses
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    rating: str  # "positive" or "negative"
    llm_call_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[dict] = None


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Record user feedback (thumbs up/down) on an LLM response.

    Links to the llm_calls table via llm_call_id so feedback can be
    correlated with the exact prompt, response, and grounding score
    that produced it.
    """
    if req.rating not in ("positive", "negative"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="rating must be 'positive' or 'negative'")

    from core.db.llm_logger import log_feedback
    feedback_id = await log_feedback(
        rating=req.rating,
        llm_call_id=uuid.UUID(req.llm_call_id) if req.llm_call_id else None,
        session_id=req.session_id,
        user_id=req.user_id,
        context=req.context,
    )
    return {"feedback_id": str(feedback_id) if feedback_id else None, "status": "ok"}


# ── Feedback analytics (read-only) ─────────────────────────────────────

@router.get("/feedback/summary")
async def feedback_summary():
    """Aggregate feedback metrics for the admin analytics dashboard.

    Returns total counts, positive/negative split, avg grounding score
    by rating, and daily trend for the last 30 days.
    """
    from core.db.postgres import get_pool
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Overall counts
        totals = await conn.fetchrow("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE rating = 'positive') AS positive,
                COUNT(*) FILTER (WHERE rating = 'negative') AS negative
            FROM feedback
        """)

        # Grounding score by rating (joined with llm_calls)
        by_rating = await conn.fetch("""
            SELECT
                f.rating,
                COUNT(*) AS count,
                AVG(lc.grounding_score)::float AS avg_grounding,
                AVG(lc.duration_ms)::int AS avg_duration_ms
            FROM feedback f
            LEFT JOIN llm_calls lc ON lc.id = f.llm_call_id
            WHERE lc.id IS NOT NULL
            GROUP BY f.rating
        """)

        # 30-day trend
        trend = await conn.fetch("""
            SELECT
                date_trunc('day', created_at)::date AS day,
                COUNT(*) FILTER (WHERE rating = 'positive') AS positive,
                COUNT(*) FILTER (WHERE rating = 'negative') AS negative
            FROM feedback
            WHERE created_at >= NOW() - INTERVAL '30 days'
            GROUP BY day
            ORDER BY day
        """)

        # Top flagged responses — recent negatives with full context
        recent_negatives = await conn.fetch("""
            SELECT f.id, f.rating, f.created_at, f.user_id,
                   lc.caller, lc.grounding_score, lc.duration_ms,
                   substring(lc.prompt, 1, 200) AS prompt_preview,
                   substring(lc.response, 1, 200) AS response_preview
            FROM feedback f
            LEFT JOIN llm_calls lc ON lc.id = f.llm_call_id
            WHERE f.rating = 'negative'
            ORDER BY f.created_at DESC
            LIMIT 10
        """)

    total = totals["total"] or 0
    positive = totals["positive"] or 0
    negative = totals["negative"] or 0
    positive_rate = (positive / total) if total > 0 else 0.0

    return {
        "totals": {
            "total": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive_rate, 3),
        },
        "by_rating": [dict(r) for r in by_rating],
        "trend": [
            {"day": str(r["day"]), "positive": r["positive"], "negative": r["negative"]}
            for r in trend
        ],
        "recent_negatives": [
            {**dict(r), "id": str(r["id"]), "created_at": r["created_at"].isoformat()}
            for r in recent_negatives
        ],
    }


@router.get("/feedback/export")
async def feedback_export(rating: Optional[str] = None):
    """Export feedback records as CSV for offline review.

    Args:
        rating: Optional filter — 'positive' or 'negative'

    Returns a CSV file with feedback rows joined to their LLM calls.
    Columns: created_at, rating, caller, model, temperature, duration_ms,
    grounding_score, user_id, session_id, prompt, response.
    """
    import csv
    import io
    from fastapi.responses import StreamingResponse

    from core.db.postgres import get_pool
    pool = await get_pool()

    where_clauses = []
    params: list[Any] = []
    idx = 1
    if rating in ("positive", "negative"):
        where_clauses.append(f"f.rating = ${idx}")
        params.append(rating)
        idx += 1
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT f.created_at, f.rating, f.user_id, f.session_id,
                   lc.caller, lc.model, lc.temperature, lc.duration_ms,
                   lc.grounding_score, lc.prompt, lc.response
            FROM feedback f
            LEFT JOIN llm_calls lc ON lc.id = f.llm_call_id
            {where_sql}
            ORDER BY f.created_at DESC
            """,
            *params,
        )

    # Build CSV in memory
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow([
        "created_at", "rating", "user_id", "session_id",
        "caller", "model", "temperature", "duration_ms",
        "grounding_score", "prompt", "response",
    ])
    for r in rows:
        writer.writerow([
            r["created_at"].isoformat() if r["created_at"] else "",
            r["rating"] or "",
            r["user_id"] or "",
            r["session_id"] or "",
            r["caller"] or "",
            r["model"] or "",
            r["temperature"] if r["temperature"] is not None else "",
            r["duration_ms"] if r["duration_ms"] is not None else "",
            r["grounding_score"] if r["grounding_score"] is not None else "",
            r["prompt"] or "",
            r["response"] or "",
        ])

    buf.seek(0)
    filename_suffix = f"-{rating}" if rating else ""
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="noesis-feedback{filename_suffix}.csv"',
        },
    )


@router.get("/feedback/recent")
async def feedback_recent(limit: int = 50, rating: Optional[str] = None):
    """List recent feedback records with joined LLM call context.

    Args:
        limit: Max rows to return (default 50, max 500)
        rating: Optional filter — 'positive' or 'negative'

    Each row includes the prompt, response, grounding score, model,
    and duration from the linked llm_calls row.
    """
    from core.db.postgres import get_pool
    pool = await get_pool()
    limit = min(limit, 500)

    where_clauses = []
    params: list[Any] = []
    idx = 1

    if rating in ("positive", "negative"):
        where_clauses.append(f"f.rating = ${idx}")
        params.append(rating)
        idx += 1

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(limit)
    limit_param = f"${idx}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT f.id, f.rating, f.created_at, f.user_id, f.session_id,
                   f.context,
                   lc.id AS llm_call_id,
                   lc.caller, lc.model, lc.temperature, lc.duration_ms,
                   lc.grounding_score, lc.prompt, lc.response
            FROM feedback f
            LEFT JOIN llm_calls lc ON lc.id = f.llm_call_id
            {where_sql}
            ORDER BY f.created_at DESC
            LIMIT {limit_param}
            """,
            *params,
        )

    return [
        {
            **{k: v for k, v in dict(r).items() if k not in ("id", "llm_call_id", "created_at")},
            "id": str(r["id"]),
            "llm_call_id": str(r["llm_call_id"]) if r["llm_call_id"] else None,
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]
