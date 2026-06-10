"""Requirements → Model generation endpoint (Agent Atlas DIAG-37, option b).

Turns a requirements document into a FIRST-CUT Agent Atlas model. This is the only
place in Axiom Core that calls a cloud (Anthropic) model, and ONLY for this non-PHI
architecture-reasoning task — clinical work stays on local inference. The API key
lives in .env (untracked) and is read via core.config.settings; it is never logged
and never returned to the client.

Contract (matches agent-atlas-studio docs/llm-provider-integration.md §2):
    POST /generate-model   { "requirements": "<text>", "options": {...} }
    200                    { "model": { "objects": {...}, "edges": [...],
                                        "subjectAreas": [...] }, "notes": [...] }

The model is a DRAFT TO REFINE — the studio normalizes + validates it and the human
ratifies it. We do not pretend it's authoritative (intent isn't fully in the doc).
"""

from __future__ import annotations

import asyncio
import json as json_mod
import logging
import time
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import settings

log = logging.getLogger("noesis.generate")

router = APIRouter(tags=["generate"])

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class GenerateRequest(BaseModel):
    requirements: str
    options: dict | None = None


# ── Async job store ─────────────────────────────────────────────────────────
# A full-doc generation takes ~2-3 minutes — longer than a WebKit/Tauri webview
# will hold a single fetch open (~60s), so the studio can't wait on one request.
# Instead POST enqueues a job and returns immediately; the studio polls GET
# /generate-model/{id} (fast calls, never near the webview cap) until it's done.
# Single-process uvicorn → a plain in-memory dict is sufficient; jobs are
# ephemeral (a draft you either apply or regenerate) so no persistence is needed.
_JOBS: dict[str, dict] = {}
_JOB_TTL_SECONDS = 3600  # reap finished jobs after an hour so the dict can't grow unbounded


def _reap_jobs() -> None:
    now = time.monotonic()
    stale = [jid for jid, j in _JOBS.items()
             if j.get("status") in ("done", "error") and now - j.get("finished_at", now) > _JOB_TTL_SECONDS]
    for jid in stale:
        _JOBS.pop(jid, None)


# The contract + rules the model must follow. Kept verbatim-strict so the output
# normalizes cleanly in the studio. Mirrors docs/architecture/object-semantics.md.
SYSTEM_PROMPT = """\
You are an architecture assistant for Agent Atlas, an Erwin-style modeler for agentic AI \
platforms. Given a requirements document, you produce a FIRST-CUT model as JSON. It is a \
draft a human will refine — propose a sound structure, do not invent precision the document \
doesn't contain, and flag low-confidence inferences in `notes`.

Return ONLY a JSON object with this exact shape (no prose, no markdown fences):
{
  "model": {
    "objects": { "<id>": { "id": "<id>", "kind": "<kind>", "parent": "<parent-id|null>",
                           "data": { ...kind-specific fields... } }, ... },
    "edges": [ { "id": "e-<source>-<target>", "source": "<id>", "target": "<id>" }, ... ],
    "subjectAreas": [ { "id": "sa-<x>", "name": "<view name>", "memberIds": ["<id>",...], "hiddenIds": [] }, ... ]
  },
  "notes": [ "low-confidence inference or assumption the human should confirm", ... ]
}

The seven object KINDS and the hierarchy (parent rules — obey strictly):
- orchestrator: the single root control plane. parent = null. Exactly one. data: {id, owner, version, description, controlFlow ("state-machine"|"dag"|"sequential"), stateStore}
- task: a stage of the workflow. parent = orchestrator or another task. data: {id, label, description}
- agent: a single-responsibility worker that REASONS with an LLM. parent = a task. data: {id, owner, version, responsibility, model:{provider,name,pinned}, refusalConditions:[...], refusalEmits, telemetry:[{name,attributes:[...]}]}
- tool: a deterministic typed call boundary (no LLM). parent = an agent. data: {id, owner, version, description, effect ("read"|"write"|"external"), authScope}
- job: async/long-running work. parent = an agent (agent-dispatched) OR a task (workflow-dispatched). data: {id, owner, version, description, queue, trigger, timeoutSeconds, retries}
- system: a datastore/external system that is TOUCHED. parent = an agent, or a task if shared across the stage. data: {id, owner, version, description, systemKind ("relational-db"|"document-store"|"vector-store"|"graph-db"|"external-api"|"state-store"), connection, authScope}
- router: dynamic model selection for ONE agent. parent = that agent. data: {id, owner, version, description, candidates:[{provider,name,pinned}], optimizeFor:[...], rules:[{when,select}], fallback}

Rules:
- ids are lowercase-hyphenated, UNIQUE across the whole model, and stable (they are the manifest filename + identity). data.id must equal the object id.
- Only Task sits directly under the orchestrator. Tools/jobs/systems/routers are leaves (no children). An agent reasons; if something does no reasoning, it's a tool or job, not an agent.
- edges connect an agent to the tools/jobs/routers/systems it uses (source=agent id, target=the used object id).
- Keep agents single-responsibility (no "and" in the responsibility).
- Prefer a complete, valid skeleton over guessed detail. Put assumptions in `notes`.

COMPLETENESS — this is critical. Model EVERYTHING the document describes; do not summarize \
or omit. Walk the document and create an object for every stage/task, every agent, every \
tool, every system (every named source AND target), every job, and every router it mentions \
— even if it lists many similar ones (e.g. multiple source databases or target platforms, \
model each as its own system). If the document names six pipeline stages, your model has six \
tasks; if it names eight systems, your model has eight systems. Dropping or merging items the \
document calls out is a failure. Before finishing, re-read the document and confirm each \
named element appears in your model; if you had to leave something out (e.g. to fit length), \
say so explicitly in `notes`.
"""


@router.post("/generate-model")
async def generate_model(req: GenerateRequest):
    """Enqueue a generation job and return its id immediately (202).

    The studio polls GET /generate-model/{job_id} for the result. We return fast so
    the webview's request timeout never trips on the multi-minute Claude call.
    """
    if not settings.model_gen_enabled:
        raise HTTPException(status_code=503, detail="Model generation is disabled on this server.")
    if not settings.anthropic_api_key.get_secret_value():
        raise HTTPException(status_code=503, detail="No ANTHROPIC_API_KEY configured on the server.")
    text = (req.requirements or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty requirements.")

    _reap_jobs()
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {"status": "pending", "started_at": time.monotonic()}
    asyncio.create_task(_run_job(job_id, text))
    log.info("generate-model: enqueued job %s (%d chars)", job_id, len(text))
    # 202 Accepted + job id → the studio knows to poll. (A synchronous adopter
    # backend would return 200 with {model,notes} instead; the studio handles both.)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "pending"})


@router.get("/generate-model/{job_id}")
async def generate_model_status(job_id: str):
    """Poll a generation job. Returns {status, model?, notes?, error?}."""
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job id.")
    if job["status"] == "pending":
        return {"status": "pending"}
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error", "Generation failed.")}
    return {"status": "done", "model": job["model"], "notes": job["notes"]}


async def _run_job(job_id: str, text: str) -> None:
    """Background worker: run the generation and stash the result on the job."""
    try:
        model, notes = await _generate(text)
        _JOBS[job_id] = {"status": "done", "model": model, "notes": notes, "finished_at": time.monotonic()}
        log.info("generate-model: job %s done (%d objects)", job_id, len((model or {}).get("objects", {})))
    except HTTPException as exc:
        _JOBS[job_id] = {"status": "error", "error": str(exc.detail), "finished_at": time.monotonic()}
        log.error("generate-model: job %s failed: %s", job_id, exc.detail)
    except Exception as exc:  # never let a worker crash silently
        _JOBS[job_id] = {"status": "error", "error": f"Unexpected error: {type(exc).__name__}", "finished_at": time.monotonic()}
        log.error("generate-model: job %s crashed: %s", job_id, exc, exc_info=True)


async def _generate(text: str) -> tuple[dict, list[str]]:
    """The actual Claude call + parse. Returns (model, notes). Raises HTTPException
    on upstream/timeout errors; degrades to an empty-but-valid model on parse failure."""
    key = settings.anthropic_api_key.get_secret_value()
    payload = {
        "model": settings.model_gen_model,
        "max_tokens": settings.model_gen_max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Requirements document:\n\n{text}\n\nProduce the first-cut model JSON now."}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    # A full model (40-50 objects) can take a few minutes to generate, so give the
    # upstream call generous headroom — far longer than the small-doc case needs.
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.model_gen_timeout, connect=15.0)) as client:
            resp = await client.post(ANTHROPIC_URL, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        log.error("generate-model: upstream request failed: %s", exc)
        raise HTTPException(status_code=504, detail=f"The model provider did not respond in time ({type(exc).__name__}). Try a shorter requirements doc or raise MODEL_GEN_TIMEOUT.")

    if resp.status_code != 200:
        # Don't leak the key; surface a trimmed upstream message.
        body = resp.text[:300]
        log.error("generate-model: provider %s: %s", resp.status_code, body)
        raise HTTPException(status_code=502, detail=f"Model provider error ({resp.status_code}).")

    data = resp.json()
    truncated = data.get("stop_reason") == "max_tokens"  # ran out of output budget
    # Anthropic Messages API: content is a list of blocks; take the text.
    try:
        out_text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        parsed = json_mod.loads(_strip_fences(out_text))
    except (ValueError, AttributeError) as exc:
        log.error("generate-model: could not parse model JSON (truncated=%s): %s", truncated, exc)
        msg = ("The model response was cut off (hit the token limit) before valid JSON completed — "
               "raise MODEL_GEN_MAX_TOKENS or shorten the requirements."
               if truncated else
               "The generator did not return valid model JSON; nothing was applied. Try again.")
        return {"objects": {}, "edges": [], "subjectAreas": []}, [msg]

    model = parsed.get("model", parsed)
    notes = parsed.get("notes", []) if isinstance(parsed.get("notes"), list) else []
    if truncated:
        notes.append("⚠ The response was truncated at the token limit — the model may be incomplete. "
                     "Raise MODEL_GEN_MAX_TOKENS and regenerate for a full model.")
    return model, notes


def _strip_fences(s: str) -> str:
    """Tolerate a model that wraps JSON in ```json fences despite instructions."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()
