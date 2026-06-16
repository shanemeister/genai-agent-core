"""Codebase scan → recovered declaration (Agent Atlas Reverse Engineering, R1).

The INVERSE of the requirements→declaration generator: this reads the EXISTING
Noesis source tree and recovers the structural skeleton of the agentic estate —
the MCP tool servers + their tools, the datastore/LLM systems the code touches,
the worker modules (the clinical validation pipeline), and the call edges between
them. It does NOT invent governance: grounding thresholds, prohibited actions,
data classification, residency, refusal rules are NOT in the code, so every
governance field comes back EMPTY. That gap — "here is what your code does, here
is everything about it that isn't governed" — is the point (the onboarding wedge).

Why this is reliable: it recovers STRUCTURE by parsing recognizable patterns
(ast for @mcp.tool / FastMCP, regex for db/LLM clients), not by inference. "Intent
is not in the code," so the recovered declaration is a draft a human ratifies and
governs — exactly like the generated one.

SECURITY: emits connection HINTS only (e.g. "postgresql", "neo4j-bolt"), never the
actual host/credentials it finds in config.py. The recovered declaration carries no
secrets.

Contract (same shape the studio consumes for any declaration):
    POST /scan-codebase  { "root": "<abs path | omitted = the running core/>" }
    200  { "declaration": { objects, edges, subjectAreas }, "notes": [...] }
"""

from __future__ import annotations

import ast
import logging
import os
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger("noesis.scan")
router = APIRouter(tags=["scan"])


# RE scans a LOCAL CODE WORKSPACE — the operator brings the code to the scanner
# (clone/checkout the repos that comprise the system into one workspace) and selects
# from it via the host browser (/list-dir). NO hardcoded presets: a real engagement
# has none, and deployment-specific paths in code violate the no-demo-shaped-code
# rule. Discovery is purely operator-driven (browse + type + git-URL clone later).
# BROWSE_ROOT is where the browser starts; defaults to home, fence-able via env.
BROWSE_ROOT = os.environ.get("ATLAS_SCAN_BROWSE_ROOT") or os.path.expanduser("~")


class ScanRequest(BaseModel):
    root: str | None = None    # absolute path of the repo/workspace folder to scan
    packs: list[str] | None = None  # recognizer packs to run; None/[] = ALL applicable
    # Packs are ADDITIVE, not a selector: a scan runs a SET of packs and UNIONs their
    # recovered objects/edges (dedupe collapses a system recovered by two packs). A
    # MIXED repo (e.g. an ML pipeline that grows agents/MCP) is first-class — both the
    # ml-pipeline and agentic packs contribute to ONE estate. Default (None) = run every
    # registered pack; the operator may NARROW (DIAG-55 operator-picks), and any pack
    # left out whose signals ARE present is reported LOUDLY (no silent half-coverage).


class ListDirRequest(BaseModel):
    path: str | None = None  # dir on the scanner host to list; default = BROWSE_ROOT


class CensusRequest(BaseModel):
    root: str | None = None    # absolute path of the repo/workspace folder to census


# ════════════════════════════════════════════════════════════════════════════════
# RECOGNIZER PACKS (the domain/architecture-style axis, Axis 1)
# ────────────────────────────────────────────────────────────────────────────────
# A pack is a named bundle of recognizers + its OWN gap vocabulary. Packs are
# ADDITIVE: a scan runs a set of them and unions the results. This is the seam that
# lets Atlas recover NON-agentic estates (ML pipelines, RL systems, …) without baking
# any single domain's assumptions into the engine. Each pack supplies:
#   system_fingerprints : [{id, systemKind, hint, patterns[], desc}]  — datastores/APIs
#   worker_hints        : (substr, …)   — module-name signals for a worker/job
#   worker_kind         : "agent" | "batch-job"  — what KIND of object a worker is
#   import_system_map   : [(pattern, system_id), …]  — worker→system edge inference
#   responsibility      : f(rel) -> str  — per-worker description (domain-neutral phrasing)
#   notes               : {system, worker, edge}  — DOMAIN-SPECIFIC gap language. This
#                         is where "PHI/clinical" lives for the clinical pack and
#                         "lineage/ownership/contention" lives for the ML pack — so no
#                         pack emits another domain's gaps (no PHI-cosplay on a pipeline).
# A "system" object is recovered if ANY pack's fingerprint matches. The same engine
# (_scan_systems / _scan_workers) runs once per active pack; the merge dedupes.
# ════════════════════════════════════════════════════════════════════════════════

# ── Pack: agentic-clinical (the original Noesis recognizers — UNCHANGED, so scanning
# Noesis is byte-identical to before). Recognizes MCP servers/tools + clinical
# datastores + reasoning workers over (often PHI-bearing) data. ──────────────────
AGENTIC_CLINICAL_PACK = {
    "id": "agentic-clinical",
    "scan_mcp": True,  # only this pack parses FastMCP/@mcp.tool
    "system_fingerprints": [
        {"id": "postgres", "systemKind": "relational-db", "hint": "postgresql",
         "patterns": [r"\basyncpg\b", r"create_pool", r"psycopg"], "desc": "PostgreSQL — primary store (FHIR resources, analyses, memory, documents)."},
        {"id": "neo4j", "systemKind": "graph-db", "hint": "neo4j-bolt",
         "patterns": [r"\bneo4j\b", r"AsyncGraphDatabase", r"bolt://"], "desc": "Neo4j — SNOMED/ICD ontology graph."},
        {"id": "llm-inference", "systemKind": "external-api", "hint": "openai-compatible (local)",
         "patterns": [r"11434", r"\bollama\b", r"llm_base_url", r"/v1/chat", r"ask_llm"], "desc": "Local LLM inference (Ollama, OpenAI-compatible) — on-prem, no PHI egress by design."},
        {"id": "pgvector", "systemKind": "vector-store", "hint": "pgvector",
         "patterns": [r"\bpgvector\b", r"SentenceTransformer", r"\bnomic\b", r"embedding"], "desc": "pgvector — RAG embeddings over clinical guidelines + product docs."},
        {"id": "fhir-endpoint", "systemKind": "fhir", "hint": "fhir-r4",
         "patterns": [r"\bfhir\b", r"FHIR", r"Bundle\b", r"Patient\b.*resource", r"\bsynthea\b"], "desc": "FHIR R4 source/consumer — patient records, clinical resources (likely PHI-bearing)."},
    ],
    "worker_hints": ("extractor", "validator", "mapper", "checker", "inference",
                     "analyzer", "analysis", "classifier", "resolver", "proposer",
                     "scorer", "rules", "impact", "reasoner"),
    "worker_kind": "agent",
    "import_system_map": [
        (r"\bcore\.graph\b", "neo4j"), (r"\bneo4j\b", "neo4j"), (r"AsyncGraphDatabase", "neo4j"),
        (r"\bcore\.rag\b", "pgvector"), (r"\bpgvector\b", "pgvector"), (r"SentenceTransformer", "pgvector"),
        (r"\bcore\.db\b", "postgres"), (r"\basyncpg\b", "postgres"), (r"\bpsycopg\b", "postgres"),
        (r"core\.api\.shared", "llm-inference"), (r"\bollama\b", "llm-inference"),
        (r"\bfhirclient\b", "fhir-endpoint"), (r"\bcore\.fhir\b", "fhir-endpoint"),
    ],
    "responsibility": lambda rel: f"Reasoning worker recovered from {rel}.",
    # CORROBORATION gate: only claim a file as a reasoning AGENT if this pack actually
    # found its domain present (an LLM/vector system or an MCP server). Otherwise the
    # agentic hints ("inference", "scorer", "analysis", "rules") would mislabel ML batch
    # jobs as agents in a repo that has no agents at all (the F5 ordering hazard when
    # this pack runs before ml-pipeline). A pack with no own-domain evidence yields its
    # name-matched files to a later pack (e.g. ml-pipeline claims them as batch jobs).
    "requires_corroboration": True,
    "corroborating_systems": ("llm-inference", "pgvector", "neo4j", "fhir-endpoint"),
    "notes": {
        "worker": "Recovered {n} worker module(s) by name heuristic. These reasoning units touch (often PHI-bearing) clinical data and call the LLM + ontology, but declare NO data classification, residency, grounding threshold, or refusal rule in code — the primary governance gap for a healthcare system.",
        "edge": "Inferred {n} worker→system data-flow edge(s) from worker file imports/usage (the recovered PHI/data path). Governance (classification, residency, refusal) is still UNDECLARED on these flows — each edge crosses into a datastore with no declared protection.",
    },
}

# ── Pack: ml-pipeline (tournament/supervised ML — Numerai-shaped, but domain-neutral).
# Recognizes the REAL systems of an ML estate (data lake, external submission API, GPU
# pool, tuning store, scheduler) and BATCH JOBS (not agents). Its gap vocabulary is
# lineage/ownership/contention/provenance/reproducibility — NOT PHI/refusal. ───────
ML_PIPELINE_PACK = {
    "id": "ml-pipeline",
    "scan_mcp": False,
    "system_fingerprints": [
        {"id": "data-lake", "systemKind": "data-lake", "hint": "parquet/columnar",
         "patterns": [r"read_parquet", r"\.parquet\b", r"\bpyarrow\b", r"\bfastparquet\b", r"to_parquet"],
         "desc": "Columnar data lake (parquet) — training/validation/live datasets. Data OWNERSHIP + LINEAGE are not declared in code."},
        {"id": "model-submission-api", "systemKind": "external-api", "hint": "tournament-api",
         "patterns": [r"\bnumerapi\b", r"NumerAPI", r"upload_predictions", r"download_dataset"],
         "desc": "External tournament/scoring API — OUTBOUND predictions + INBOUND datasets. Submission provenance (which model, which data round) is not declared."},
        {"id": "gpu-pool", "systemKind": "compute-resource", "hint": "cuda-gpu",
         "patterns": [r"CUDA_VISIBLE_DEVICES", r"task_type[\"']?\s*[:=]\s*[\"']?GPU", r"tree_method.*gpu", r"\bdevices\b\s*[:=]", r"device\s*=\s*[\"']cuda"],
         "desc": "Shared GPU compute pool — training accelerators. Resource governance (device assignment, contention with co-tenant workloads) is not declared in code."},
        {"id": "tuning-store", "systemKind": "experiment-store", "hint": "optuna/sqlite",
         "patterns": [r"\boptuna\b", r"create_study", r"\bstudy_name\b", r"storage\s*="],
         "desc": "Hyperparameter-tuning store (Optuna/sqlite) — trial history + best params. Reproducibility (seed, study lineage) is not declared."},
        {"id": "scheduler", "systemKind": "scheduler", "hint": "cron/systemd",
         "patterns": [r"\bschedule\b", r"\bcrontab\b", r"systemd", r"\.service\b", r"\bAPScheduler\b"],
         "desc": "Job scheduler (cron/systemd) — the de-facto control plane driving training/scoring/submission runs. Run provenance + failure handling are not declared."},
    ],
    "worker_hints": ("train", "predict", "score", "submit", "inference", "ensemble",
                     "blend", "neutralize", "feature", "cluster", "evaluate", "tune",
                     "optuna", "pipeline", "backtest"),
    "worker_kind": "batch-job",
    "import_system_map": [
        (r"\bnumerapi\b", "model-submission-api"), (r"NumerAPI", "model-submission-api"),
        (r"\boptuna\b", "tuning-store"),
        (r"read_parquet|\.parquet|to_parquet|\bpyarrow\b", "data-lake"),
        (r"\bschedule\b|APScheduler", "scheduler"),
    ],
    "responsibility": lambda rel: f"Batch job recovered from {rel}.",
    "notes": {
        "worker": "Recovered {n} batch job(s) by name heuristic. These are pipeline stages (train/score/submit/feature/ensemble), NOT reasoning agents — they declare NO data lineage, model/submission provenance, resource governance, or reproducibility contract in code. That is the governance gap for an ML pipeline (not PHI/refusal).",
        "edge": "Inferred {n} job→system data-flow edge(s) from job file imports/usage (the recovered data path: lake → train → submission API). Lineage + provenance are UNDECLARED on these flows — there is no declared record of which data round produced which submission.",
    },
}

# Pack registry. Add a domain by appending a pack here — the engine is untouched.
# (Future: rl-pipeline — environment/policy/reward/replay-buffer/online-loop with its
# own gap vocab: reward hacking, exploration safety, sim-vs-live shift, off-policy
# provenance. Drops in as another entry; census flags it if a scan runs without it.)
PACKS = {p["id"]: p for p in (AGENTIC_CLINICAL_PACK, ML_PIPELINE_PACK)}


def _select_packs(requested: list[str] | None) -> list[dict]:
    """Resolve the requested pack ids to pack dicts. None/[] = ALL registered packs
    (additive default). Unknown ids are ignored (the caller is told via a note)."""
    if not requested:
        return list(PACKS.values())
    return [PACKS[p] for p in requested if p in PACKS]


def _py_files(root: str):
    """All .py files under root, skipping caches/venvs/build dirs AND vendored/nested
    repos. Layout-agnostic so it works on any repo structure (Axiom core/, Gateway
    src/noesis_gateway/, …).

    F2 fix: a directory that is ITSELF a git checkout (contains a .git entry) is a
    VENDORED third-party repo (e.g. a cloned benchmark) — first-party recovery must
    NOT descend into it, or it pollutes the estate with someone else's code. We skip
    such subdirs but never the scan ROOT itself (the root is legitimately a repo)."""
    SKIP = {"__pycache__", ".git", ".venv", "venv", "node_modules", "build", "dist", ".mypy_cache", ".pytest_cache"}
    self_path = os.path.abspath(__file__)  # the scanner's own source
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        # Drop any CHILD that is itself a git checkout (a vendored/nested repo) — its
        # `.git` may be a directory OR a file (submodule/worktree gitdir pointer). We
        # check the CHILDREN here, not dirpath, so a vendored repo that sits as a direct
        # child of the scan root is still excluded; the root itself is never a child of
        # itself, so it is always kept. (The earlier bug guarded on `dirpath != root`,
        # which skipped the check for every direct child of the root — exactly where a
        # vendored tree like tabular-benchmark/ lives.)
        dirnames[:] = [d for d in dirnames
                       if not os.path.exists(os.path.join(dirpath, d, ".git"))]
        for f in filenames:
            if not f.endswith(".py"):
                continue
            full = os.path.join(dirpath, f)
            # A static-analysis tool must NOT fingerprint its OWN recognizer table:
            # this module literally contains every system pattern as a string literal,
            # so scanning it self-matches every pack (e.g. recovering a 'data-lake' from
            # the ml-pipeline pattern text). Exclude the scanner's own source.
            if os.path.abspath(full) == self_path:
                continue
            yield full


def _scan_mcp_servers(core_dir: str, objects: dict, edges: list, notes: list) -> None:
    """Recover MCP tool servers (FastMCP) and their @mcp.tool() functions as
    Agent + Tool objects, found ANYWHERE in the tree. The server is the
    agent-equivalent reasoning host; each @mcp.tool is a deterministic Tool."""
    for path in _py_files(core_dir):
        fn = os.path.basename(path)
        try:
            src = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if "FastMCP" not in src and "@mcp.tool" not in src:
            continue  # cheap pre-filter before parsing
        try:
            tree = ast.parse(src, filename=fn)
        except SyntaxError as exc:
            notes.append(f"Could not parse {fn}: {exc}")
            continue
        server_name = None
        tools = []
        for node in ast.walk(tree):
            # mcp = FastMCP("noesis-snomed")
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fnname = getattr(node.value.func, "id", None) or getattr(node.value.func, "attr", None)
                if fnname == "FastMCP" and node.value.args:
                    a0 = node.value.args[0]
                    if isinstance(a0, ast.Constant):
                        server_name = str(a0.value)
            # @mcp.tool() async def name(...)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(target, ast.Attribute) and target.attr == "tool":
                        tools.append(node.name)
        if not server_name:
            continue
        agent_id = _slug(server_name)  # e.g. noesis-snomed
        objects[agent_id] = {
            "id": agent_id, "kind": "agent", "parent": None,
            "data": {"id": agent_id, "owner": "", "version": "1.0.0",
                     "responsibility": f"MCP server '{server_name}' — recovered from {fn}.",
                     "model": {"provider": "", "name": "", "pinned": ""},
                     "refusalConditions": [], "telemetry": [{"name": f"{agent_id}-span"}]},
        }
        for t in tools:
            tid = _slug(t)
            objects[tid] = {
                "id": tid, "kind": "tool", "parent": agent_id,
                "data": {"id": tid, "owner": "", "version": "1.0.0",
                         "description": f"MCP tool '{t}' (server {server_name}).",
                         "effect": "read", "authScope": ""},
            }
            edges.append({"id": f"e-{agent_id}-{tid}", "source": agent_id, "target": tid})
        notes.append(f"Recovered MCP server '{server_name}' with {len(tools)} tool(s) from {fn}. Governance (data class, residency, thresholds, prohibited actions) is NOT in the code — declare it.")


# Common ways code declares its inference model name — vendor-neutral, captures
# whatever value the SCANNED system uses (no hardcoded model). Ordered most-specific
# first; the first capture wins.
# Tolerate a type annotation between the name and the value (pydantic settings:
# `llm_model_name: str = "axiom-primary"`), plain assignment, and env-style.
_MODEL_NAME_PATTERNS = [
    r'llm_model_name\s*(?::[^=]+)?=\s*["\']([^"\']+)["\']',
    r'(?:^|\b)MODEL_NAME\s*=\s*["\']?([^"\'\s]+)',
    r'\bmodel_name\s*(?::[^=]+)?=\s*["\']([^"\']+)["\']',
    r'OLLAMA_MODEL\s*[:=]\s*["\']?([^"\'\s]+)',
    r'\bmodel\s*=\s*["\']([a-zA-Z0-9][\w.\-:/]+)["\']',
]


def _recover_model_name(text: str) -> str:
    """Best-effort recovery of the inference model NAME a codebase declares. Returns the
    literal string the code uses (e.g. 'axiom-primary', 'qwen2.5:32b', 'gpt-4o') or ""
    if none is found — in which case target generation leaves it as a declare-or-flag
    gap, NEVER substitutes a default. Domain/vendor-agnostic by design."""
    for pat in _MODEL_NAME_PATTERNS:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            val = m.group(1).strip()
            # skip obvious non-model captures (paths, urls, embeddings)
            if val and "/" not in val[:1] and "embed" not in val.lower() and len(val) < 80:
                return val
    return ""


def _scan_systems(core_dir: str, objects: dict, notes: list, pack: dict, _text_cache: dict) -> None:
    """Recover the datastores/external systems the code TOUCHES (by fingerprint).
    Records a connection HINT, never the real credentials. Pack-driven: uses this
    pack's system_fingerprints. _text_cache memoizes the concatenated source per root
    so running N packs reads the tree once, not N times."""
    text = _text_cache.get(core_dir)
    if text is None:
        blobs = []
        for path in _py_files(core_dir):
            try:
                blobs.append(open(path, encoding="utf-8", errors="ignore").read())
            except OSError:
                pass
        text = "\n".join(blobs)
        _text_cache[core_dir] = text
    for fp in pack["system_fingerprints"]:
        if fp["id"] in objects:
            continue  # already recovered by an earlier pack (dedupe)
        if any(re.search(p, text) for p in fp["patterns"]):
            data = {"id": fp["id"], "owner": "", "version": "1.0.0",
                    "description": fp["desc"], "systemKind": fp["systemKind"],
                    "connection": fp["hint"], "authScope": ""}  # HINT, not secrets
            # For an inference system, recover the actual model NAME the code declares
            # (vendor-neutral — whatever the scanned system uses). Lets target
            # generation pin agents to the REAL model, not a hardcoded default.
            if fp["id"] == "llm-inference":
                m = _recover_model_name(text)
                if m:
                    data["recoveredModel"] = m
            objects[fp["id"]] = {"id": fp["id"], "kind": "system", "parent": None, "data": data}
            extra = f" Model declared as '{data['recoveredModel']}'." if data.get("recoveredModel") else ""
            notes.append(f"Recovered system '{fp['id']}' ({fp['systemKind']}).{extra} Connection recorded as a hint only — no credentials emitted. Its governance (data classification, residency, ownership/lineage as applicable) is NOT declared in code.")


def _systems_from_imports(src: str, import_map: list) -> set:
    """System ids a source file reaches, inferred from its import statements only.
    Deterministic; precise (the import path names the system); follows the one hop
    that matters (worker → its data/graph/llm helper module → the datastore).
    Pack-driven: `import_map` is this pack's [(pattern, system_id)] list."""
    hits = set()
    for line in src.splitlines():
        s = line.strip()
        if not (s.startswith("from ") or s.startswith("import ")):
            continue
        for pat, sys_id in import_map:
            if re.search(pat, s):
                hits.add(sys_id)
    return hits


# Files that are NEVER a worker, regardless of pack (dataclasses, config, the MCP
# server/tool impls already captured, infra plumbing). Domain-neutral.
_WORKER_EXCLUDE = ("model", "schema", "client", "config", "util", "test", "__init__",
                   "types", "constants", "logger", "exceptions", "_server", "_tools",
                   "server", "store", "storage", "router", "routes", "_api")


def _scan_workers(core_dir: str, objects: dict, edges: list, notes: list, pack: dict) -> None:
    """Recover worker modules (the pack's WORKER_KIND — 'agent' for reasoning units,
    'batch-job' for pipeline stages) by module-name heuristic, found ANYWHERE in the
    tree so it works across repo layouts. Also infers worker→system EDGES from the
    worker's imports via the pack's import_system_map (the recovered data-flow). Edges
    are only drawn to systems actually recovered. Pack-driven throughout: hints, the
    object KIND, the responsibility phrasing, and the gap-note vocabulary all come from
    `pack` — so an ML batch job is never mislabeled an 'agent' and never gets PHI gaps."""
    worker_hints = pack["worker_hints"]
    worker_kind = pack["worker_kind"]
    import_map = pack["import_system_map"]
    describe = pack["responsibility"]
    # Corroboration gate (see pack def): if this pack requires its domain to be present
    # and none of its corroborating systems were recovered, do NOT claim workers — its
    # name hints would mislabel another domain's files. (MCP servers, recovered as
    # agents, also corroborate: an MCP server means the agentic domain is real.)
    if pack.get("requires_corroboration"):
        corroborated = (
            any(sid in objects for sid in pack.get("corroborating_systems", ()))
            or any(o["kind"] == "agent" for o in objects.values())
        )
        if not corroborated:
            return
    seen_edge = {(e["source"], e["target"]) for e in edges}
    found = 0
    edge_count = 0
    for path in _py_files(core_dir):
        stem = os.path.basename(path)[:-3]
        sl = stem.lower()
        if any(x in sl for x in _WORKER_EXCLUDE):
            continue
        if not any(h in sl for h in worker_hints):
            continue
        aid = _slug(stem)
        if aid in objects:
            continue
        rel = os.path.relpath(path, core_dir)
        # An agent carries model/refusal fields; a batch job does not (declaring those
        # on a non-reasoning stage would be the same false-shape problem as PHI gaps).
        if worker_kind == "agent":
            data = {"id": aid, "owner": "", "version": "1.0.0",
                    "responsibility": describe(rel),
                    "model": {"provider": "", "name": "", "pinned": ""},
                    "refusalConditions": [], "telemetry": [{"name": f"{aid}-span"}]}
        else:
            data = {"id": aid, "owner": "", "version": "1.0.0",
                    "responsibility": describe(rel),
                    "telemetry": [{"name": f"{aid}-span"}]}
        objects[aid] = {"id": aid, "kind": worker_kind, "parent": None, "data": data}
        found += 1
        # Infer this worker's data-flow from its IMPORTS, not raw content. Workers
        # reach datastores through named local modules (core.graph.* → neo4j,
        # core.rag.* → pgvector, core.db.postgres → postgres, core.api.shared's
        # ask_llm → llm-inference). The import PATH names the system precisely —
        # content-fingerprinting the worker's own file misses indirect access (a
        # worker that imports a graph helper never says "neo4j") and false-matches on
        # incidental mentions. One import hop is the right, deterministic signal.
        try:
            src = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            src = ""
        for sys_id in _systems_from_imports(src, import_map):
            if sys_id not in objects:           # only edge to a recovered system
                continue
            if (aid, sys_id) in seen_edge:
                continue
            edges.append({"id": f"e-{aid}-{sys_id}", "source": aid, "target": sys_id})
            seen_edge.add((aid, sys_id))
            edge_count += 1
    if found:
        notes.append(pack["notes"]["worker"].format(n=found))
    if edge_count:
        notes.append(pack["notes"]["edge"].format(n=edge_count))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")


@router.get("/scan-workspace")
async def scan_workspace():
    """Where the host folder browser starts. No presets — discovery is operator-
    driven (browse the workspace + type a path)."""
    return {"browseRoot": BROWSE_ROOT}


# Markers that suggest a directory is a code repo (so the browser can hint "scan me").
_REPO_MARKERS = (".git", "pyproject.toml", "setup.py", "package.json", "src", "core", "requirements.txt", "Cargo.toml", "go.mod")


def _detect_language(dir_path: str, children: list[str]) -> str | None:
    """Best-effort language guess for a repo folder, so the operator gets a default
    they can override. The LANGUAGE TAG routes the repo to a scanner plugin (and its
    runtime), so this is a hint, not a hard classification. Cheap: top-level markers
    + a shallow extension peek; never reads file contents."""
    cset = set(children)
    # TypeScript/JS UI — package.json + a TS/JSX signal
    if "package.json" in cset:
        if "tsconfig.json" in cset or _has_ext(dir_path, (".tsx", ".ts", ".jsx")):
            return "typescript"
        return "typescript"  # a package.json repo is JS/TS either way for our purposes
    # Python (and Spark/Databricks ride on Python)
    if cset & {"pyproject.toml", "setup.py", "requirements.txt", "Pipfile"} or _has_ext(dir_path, (".py", ".ipynb")):
        # Spark/Databricks signal → spark (still Python-hosted, but a distinct plugin)
        if _has_spark_signal(dir_path):
            return "spark"
        return "python"
    # SQL-heavy repo (migrations, DDL, stored procs)
    if _has_ext(dir_path, (".sql",)):
        return "sql"
    return None  # unknown → operator picks


def _has_ext(dir_path: str, exts: tuple[str, ...]) -> bool:
    """Shallow check: does the top level (or a src/ child) contain a file with any ext?"""
    for probe in (dir_path, os.path.join(dir_path, "src")):
        try:
            for f in os.listdir(probe):
                if f.endswith(exts):
                    return True
        except OSError:
            continue
    return False


def _has_spark_signal(dir_path: str) -> bool:
    """Databricks/Spark notebook repos: .ipynb present, or a databricks-y filename."""
    try:
        names = os.listdir(dir_path)
    except OSError:
        return False
    if any(n.endswith(".ipynb") for n in names):
        return True
    return any("databricks" in n.lower() for n in names)


@router.post("/list-dir")
async def list_dir(req: ListDirRequest):
    """List the SUBDIRECTORIES of a path the scanner can READ (its own filesystem,
    incl. any mounted/shared dirs), so the studio shows a folder browser of the code
    workspace. Read-only, dirs only — never returns file contents. Per entry: whether
    it looks like a repo + a best-effort LANGUAGE guess (the operator overrides).
    Permission/IO errors degrade to an empty list, not a 500."""
    path = os.path.abspath(req.path or BROWSE_ROOT)
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"Not a readable directory: {path}")
    entries = []
    try:
        for name in sorted(os.listdir(path)):
            if name.startswith(".") and name != ".git":
                continue  # hide dotfiles (but a .git presence flags a repo below)
            full = os.path.join(path, name)
            if not os.path.isdir(full):
                continue
            try:
                children = os.listdir(full)
            except OSError:
                children = []
            is_repo = any(m in children for m in _REPO_MARKERS)
            entries.append({"name": name, "path": full, "isRepo": is_repo,
                            "language": _detect_language(full, children) if is_repo else None})
    except OSError as exc:
        log.warning("list-dir(%s): %s", path, exc)
    parent = os.path.dirname(path.rstrip("/")) if path not in ("/", "") else None
    return {"path": path, "parent": parent, "entries": entries}


@router.post("/scan-codebase")
async def scan_codebase(req: ScanRequest):
    # The operator supplies the repo/workspace folder to scan (via the host browser
    # or a typed path). No presets — discovery is operator-driven.
    if not req.root:
        raise HTTPException(status_code=400, detail="No scan path provided. Browse the workspace or type a repo path.")
    core_dir = req.root
    if not os.path.isdir(core_dir):
        raise HTTPException(status_code=400, detail=f"Scan root is not a directory: {core_dir}")

    active_packs = _select_packs(req.packs)
    if not active_packs:
        raise HTTPException(status_code=400, detail=f"No known packs in {req.packs}. Known: {list(PACKS)}.")

    objects: dict = {}
    edges: list = []
    notes: list = [
        "Recovered declaration (Agent Atlas Reverse Engineering). This is what the CODE does — the structural skeleton. "
        "Governance was not recovered because it is not in the code: data classification, residency, resource/quality "
        "constraints, prohibited actions, and compliance regimes are DECLARED decisions, not facts in the source — "
        "regardless of domain. Ratify this skeleton, then declare the governance — the gaps are the work.",
    ]
    notes.append(f"Recognizer packs run: {', '.join(p['id'] for p in active_packs)}.")

    # Run every active pack additively. Systems must be recovered BEFORE workers across
    # ALL packs, so a worker recovered by pack B can edge to a system recovered by pack
    # A (dedupe in _scan_systems keeps a shared system single). Order: MCP (gated) →
    # systems (all packs) → workers (all packs).
    _text_cache: dict = {}
    for pack in active_packs:
        if pack.get("scan_mcp"):
            _scan_mcp_servers(core_dir, objects, edges, notes)
    for pack in active_packs:
        _scan_systems(core_dir, objects, notes, pack, _text_cache)
    for pack in active_packs:
        _scan_workers(core_dir, objects, edges, notes, pack)

    # HONESTY (pack census): if the operator NARROWED the packs, check whether a pack
    # they LEFT OUT has signals present in the code. A narrowed scan that silently
    # missed half an estate is the false-coverage failure DIAG-59 exists to prevent —
    # so we report it loudly (never silently). Same contract as the language census.
    skipped = [p for p in PACKS.values() if p not in active_packs]
    if skipped:
        full_text = _text_cache.get(core_dir, "")
        for sp in skipped:
            sig = [fp["id"] for fp in sp["system_fingerprints"]
                   if any(re.search(pat, full_text) for pat in fp["patterns"])]
            if sig:
                notes.append(
                    f"⚠ COVERAGE GAP: the '{sp['id']}' pack was NOT run, but its signals ARE present "
                    f"in the code ({', '.join(sig)}). Those objects were NOT recovered — re-scan including "
                    f"the '{sp['id']}' pack for full coverage. (This scan is partial by operator choice.)"
                )

    # Tree root for layout. Only the AGENTIC pack's missing control plane warrants the
    # "inferred-orchestrator + ATLAS-Baseline" finding — a batch-only ML estate's control
    # plane is its scheduler (recovered as a system), so asserting it "needs an
    # orchestrator" would be a false, domain-wrong finding. So: if any AGENTS exist, use
    # the inferred-orchestrator + agentic finding; if the estate is batch-jobs only, use
    # a neutral synthetic root and DO NOT emit the orchestrator recommendation.
    if objects:
        agent_n = sum(1 for o in objects.values() if o["kind"] == "agent")
        if agent_n:
            orch_id = "inferred-orchestrator"
            objects[orch_id] = {
                "id": orch_id, "kind": "orchestrator", "parent": None,
                "data": {"id": orch_id, "owner": "", "version": "1.0.0",
                         "_inferred": True,  # NOT recovered from code — synthetic root
                         "description": "INFERRED root — the scanned code has NO control plane (no orchestrator/StateGraph; agents are wired ad-hoc through API routes). Atlas inserted this node only so the estate renders as a tree. It does not exist in the code.",
                         "controlFlow": "", "stateStore": ""},
            }
            for o in objects.values():
                if o["kind"] != "orchestrator" and not o.get("parent"):
                    o["parent"] = orch_id
            notes.append(
                f"STRUCTURAL FINDING: the code has NO control plane — {agent_n} agent(s) run un-orchestrated, "
                f"wired ad-hoc (no LangGraph/StateGraph, no single coordinator). The 'inferred-orchestrator' node "
                f"is a SYNTHETIC placeholder Atlas inserted for tree layout; it does not exist in the code and no "
                f"control edges were drawn to it. Governance consequence: with no control plane there is no single "
                f"point that enforces refusal, audit, or human-approval across the agents — each must guarantee these "
                f"independently. «ATLAS BASELINE recommends a root orchestrator (best practice, NOT a regulatory "
                f"requirement) — ratify to adopt, or declare the ad-hoc wiring as intentional.»"
            )
        else:
            # Batch-job-only estate: neutral layout root, no agentic control-plane claim.
            root_id = "recovered-estate"
            objects[root_id] = {
                "id": root_id, "kind": "orchestrator", "parent": None,
                "data": {"id": root_id, "owner": "", "version": "1.0.0",
                         "_inferred": True,
                         "description": "INFERRED layout root — no agents in this estate; the recovered scheduler/cron is the de-facto control plane. This node exists only so the estate renders as a tree.",
                         "controlFlow": "", "stateStore": ""},
            }
            for o in objects.values():
                if o["kind"] != "orchestrator" and not o.get("parent"):
                    o["parent"] = root_id

    return {
        "declaration": {"objects": objects, "edges": edges, "subjectAreas": []},
        "notes": notes,
        "scannedRoot": core_dir,
    }


# Directories never worth counting (vendored/build/cache) — keeps the census fast and
# the coverage % honest (a node_modules full of JS shouldn't read as "the system").
_CENSUS_SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", "out", ".next", "coverage",
    ".turbo", "target", "venv", ".venv", "__pycache__", "vendor", ".mypy_cache",
}
_CENSUS_MAX_FILES = 50000  # bounded; truncation is REPORTED, never silent.


@router.post("/census")
async def census(req: CensusRequest):
    """Count source files by extension under a root (DIAG-59 language coverage census).

    Returns RAW extension counts; the studio maps extension→language (one source of
    truth for the language map). This is the no-false-coverage safety layer: the
    studio cross-references these counts against which languages a scanner actually
    ran, and reports any DETECTED-BUT-UNSCANNED language loudly. Reads nothing but
    filenames — no file contents, no secrets. Bounded + reports truncation."""
    root = req.root
    if not root or not os.path.isdir(root):
        raise HTTPException(status_code=400, detail=f"Census root is not a directory: {root}")
    counts: dict[str, int] = {}
    total = 0
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        # prune skip/hidden dirs in place
        dirnames[:] = [d for d in dirnames if d not in _CENSUS_SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if total >= _CENSUS_MAX_FILES:
                truncated = True
                break
            lower = fn.lower()
            if lower == "dockerfile" or lower == "makefile":
                key = lower  # name-based (the studio map handles these)
            else:
                dot = lower.rfind(".")
                if dot < 0:
                    continue
                key = lower[dot + 1:]
            counts[key] = counts.get(key, 0) + 1
            total += 1
        if truncated:
            break
    return {"counts": counts, "total": total, "truncated": truncated, "censusedRoot": root}
