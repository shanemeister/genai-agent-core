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

import json as json_mod
import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.config import settings

log = logging.getLogger("noesis.generate")

router = APIRouter(tags=["generate"])

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class GenerateRequest(BaseModel):
    requirements: str
    options: dict | None = None


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
"""


@router.post("/generate-model")
async def generate_model(req: GenerateRequest):
    if not settings.model_gen_enabled:
        raise HTTPException(status_code=503, detail="Model generation is disabled on this server.")
    key = settings.anthropic_api_key.get_secret_value()
    if not key:
        raise HTTPException(status_code=503, detail="No ANTHROPIC_API_KEY configured on the server.")
    text = (req.requirements or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty requirements.")

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

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(ANTHROPIC_URL, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        log.error("generate-model: upstream request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not reach the model provider.")

    if resp.status_code != 200:
        # Don't leak the key; surface a trimmed upstream message.
        body = resp.text[:300]
        log.error("generate-model: provider %s: %s", resp.status_code, body)
        raise HTTPException(status_code=502, detail=f"Model provider error ({resp.status_code}).")

    data = resp.json()
    # Anthropic Messages API: content is a list of blocks; take the text.
    try:
        out_text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        parsed = json_mod.loads(_strip_fences(out_text))
    except (ValueError, AttributeError) as exc:
        log.error("generate-model: could not parse model JSON: %s", exc)
        # Return an empty-but-valid envelope so the studio degrades gracefully.
        return {"model": {"objects": {}, "edges": [], "subjectAreas": []},
                "notes": ["The generator did not return valid model JSON; nothing was applied. Try again or simplify the requirements."]}

    model = parsed.get("model", parsed)
    notes = parsed.get("notes", []) if isinstance(parsed.get("notes"), list) else []
    return {"model": model, "notes": notes}


def _strip_fences(s: str) -> str:
    """Tolerate a model that wraps JSON in ```json fences despite instructions."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()
