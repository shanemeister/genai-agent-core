"""Noesis LLM Inference Server — vLLM proxy for DeepSeek-R1-Distill-Llama-70B.

Replaces the old HuggingFace Transformers main.py.
vLLM runs as a separate process (see launch_vllm.sh) and exposes an
OpenAI-compatible API on port 8081. This server proxies /ask requests
to vLLM and keeps backward compatibility with the Noesis services
backend on port 8008.

Usage:
    uvicorn core.api.main_vllm:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

import io
import os
import re
import time
from typing import List, Literal, Optional

import asyncio
import json as json_mod

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VLLM_BASE = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8081")
VLLM_MODEL = os.getenv("VLLM_MODEL_NAME", "./models/deepseek-r1-70b-w4a16")
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Noesis LLM Proxy", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.relufox.ai",
        "http://192.168.4.25:3000",
        "http://localhost:3000",
        "http://192.168.4.22:1420",
        "tauri://localhost",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas (backward-compatible with the old /ask endpoint)
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., description="Your question / prompt")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2000, ge=50, le=8000)


class AskResponse(BaseModel):
    answer: str
    model: str
    processing_time: float


# ---------------------------------------------------------------------------
# vLLM proxy helper
# ---------------------------------------------------------------------------
async def call_vllm(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> dict:
    """Call the vLLM OpenAI-compatible chat/completions endpoint.

    Uses the chat format so DeepSeek-R1's chat template is applied,
    which triggers proper <think>...</think> reasoning tags.
    """
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
        resp = await client.post(
            f"{VLLM_BASE}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]["message"]["content"]
    model_name = data.get("model", VLLM_MODEL)
    # Clean up model name for display
    display_name = "DeepSeek-R1-70B"
    return {"answer": choice.strip(), "model": display_name}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Noesis LLM Proxy — vLLM backend", "model": VLLM_MODEL}


@app.get("/health")
async def health():
    """Check vLLM is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{VLLM_BASE}/health")
            vllm_ok = resp.status_code == 200
    except Exception:
        vllm_ok = False

    return {
        "status": "ok" if vllm_ok else "degraded",
        "model": VLLM_MODEL,
        "vllm_reachable": vllm_ok,
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Backward-compatible /ask endpoint — proxies to vLLM."""
    start = time.time()

    try:
        result = await call_vllm(
            prompt=req.question,
            temperature=req.temperature or 0.7,
            max_tokens=req.max_tokens or 2000,
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"vLLM error: {e.response.text}")
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="vLLM server not reachable. Is launch_vllm.sh running?",
        )

    elapsed = round(time.time() - start, 2)

    return AskResponse(
        answer=result["answer"],
        model=result["model"],
        processing_time=elapsed,
    )


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------
@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Stream tokens from vLLM using SSE.

    Each SSE event is one of:
      event: token   data: {"token": "..."}
      event: done    data: {"model": "...", "processing_time": ...}
      event: error   data: {"detail": "..."}
    """
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": req.question}],
        "temperature": req.temperature or 0.7,
        "max_tokens": req.max_tokens or 2000,
        "stream": True,
    }

    async def generate():
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{VLLM_BASE}/v1/chat/completions",
                    json=payload,
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
                        content = delta.get("content", "")
                        if content:
                            yield f"event: token\ndata: {json_mod.dumps({'token': content})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json_mod.dumps({'detail': str(e)})}\n\n"
            return

        elapsed = round(time.time() - start, 2)
        yield f"event: done\ndata: {json_mod.dumps({'model': 'DeepSeek-R1-70B', 'processing_time': elapsed})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Document processing (FoxDigest compatibility)
# ---------------------------------------------------------------------------

def _split_thinking(raw: str) -> tuple[str, Optional[str]]:
    """Strip <think> reasoning from DeepSeek-R1 output."""
    match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        answer = raw[: match.start()] + raw[match.end() :]
        return answer.strip(), reasoning or None
    if "</think>" in raw:
        parts = raw.split("</think>", 1)
        return (parts[1].strip() if len(parts) > 1 else ""), parts[0].strip() or None
    return raw.strip(), None


def extract_text_from_file(filename: str, content: bytes) -> str:
    """Extract text from uploaded document."""
    ext = filename.lower()

    if ext.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(content))
        text = "\n".join(para.text for para in doc.paragraphs)

    elif ext.endswith(".pdf"):
        import fitz
        with fitz.open(stream=content, filetype="pdf") as pdf:
            text = "\n".join(page.get_text() for page in pdf)

    elif ext.endswith((".txt", ".md")):
        text = content.decode(errors="ignore")

    elif ext.endswith((".png", ".jpg", ".jpeg")):
        from PIL import Image
        import pytesseract
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image, lang="eng")

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    if len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Unable to extract usable text from document.")

    return text.strip()


@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """Summarize an uploaded document using DeepSeek-R1 via vLLM.

    Backward-compatible with FoxDigest's localLLM.js client.
    Accepts: PDF, DOCX, TXT, MD, PNG, JPG.
    Returns: {summary, key_points, processing_time, model_used}
    """
    try:
        start = time.time()
        content = await file.read()

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")

        extracted_text = extract_text_from_file(file.filename, content)

        # Truncate to fit context window (~3000 tokens of text, leave room for prompt + response)
        max_chars = 12000
        truncated_text = extracted_text[:max_chars]

        # Adjust summary length based on input
        if len(truncated_text) < 1000:
            length_instruction = "in under 75 words"
        elif len(truncated_text) < 3000:
            length_instruction = "in 100-150 words"
        else:
            length_instruction = "in 200-250 words"

        prompt = (
            f"You are a professional summarization assistant. Summarize the following document clearly "
            f"and concisely, {length_instruction}, using Markdown.\n\n"
            "Use this format:\n\n"
            "**Overview**\n[Brief description]\n\n"
            "**Details**\n[2-4 sentence explanation]\n\n"
            "**Outcome**\n[Optional: result, insight, or next steps.]\n\n"
            "### Key Points\n"
            "- [Point 1]\n- [Point 2]\n- [Point 3]\n\n"
            "Do not explain your response. Return only formatted markdown.\n\n"
            f"Document:\n{truncated_text}"
        )

        result = await call_vllm(prompt=prompt, temperature=0.3, max_tokens=1000)

        # Strip <think> reasoning from DeepSeek output
        raw_answer = result["answer"]
        visible_answer, _ = _split_thinking(raw_answer)

        # Clean up common artifacts
        markdown_output = visible_answer.strip()

        # Extract key points (handle ### Key Points, **Key Points**, etc.)
        key_points_match = re.search(r"(?:#{1,3}\s*)?(?:\*\*)?Key Points(?:\*\*)?\s*\n((?:- .+\n?)+)", markdown_output)
        if key_points_match:
            key_points = [line.strip("- ").strip() for line in key_points_match.group(1).splitlines() if line.strip()]
        else:
            key_points = []

        elapsed = round(time.time() - start, 2)

        return JSONResponse(
            content={
                "summary": markdown_output,
                "key_points": key_points,
                "processing_time": elapsed,
                "model_used": "DeepSeek-R1-70B",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
