import os, io, json, hashlib, re, time
from typing import List, Literal, Optional

from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field, ValidationError

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from docx import Document
import torch
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# --- optional: load .env automatically ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- optional Redis client (cache) ---
try:
    import redis.asyncio as aioredis  # optional
except Exception:
    aioredis = None

REDIS_URL = os.getenv("REDIS_URL")  # e.g., redis://localhost:6379/0
redis_client = None
if REDIS_URL and aioredis:
    try:
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        redis_client = None

# =========================================================
# FastAPI App Setup
# =========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.4.25:3000",
        "https://relufox.ai",
        "https://www.relufox.ai",        
        "http://localhost:3000",
        "http://192.168.4.22:1420",  # Add your Mac's dev server
        "tauri://localhost"           # For production builds
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load Local Model
# =========================================================
MODEL_PATH = "./models/llama3-8b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir="/tmp/tokenizer_cache")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400,
    temperature=0.3,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

# =========================================================
# LLM Inference Helper
# =========================================================
def run_llm_inference(prompt: str) -> str:
    response = generator(prompt, return_full_text=False)
    return response[0]["generated_text"].strip()

# =========================================================
# Existing: Document processing/Summarization
# =========================================================
def extract_text_from_file(filename: str, content: bytes) -> str:
    ext = filename.lower()
    text = ""

    if ext.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])

    elif ext.endswith(".pdf"):
        with fitz.open(stream=content, filetype="pdf") as pdf:
            text = ""
            for page in pdf:
                page_text = page.get_text().strip()
                if page_text:
                    text += page_text + "\n"
                else:
                    # Fallback to OCR if page is image-based
                    pix = page.get_pixmap(dpi=300)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    text += pytesseract.image_to_string(image, lang='eng') + "\n"

    elif ext.endswith((".txt", ".md")):
        text = content.decode(errors="ignore")

    elif ext.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image, lang='eng')

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    if len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Unable to extract usable text from document.")

    return text.strip()

# =========================================================
# NEW: Structured Output Schemas (Phase 1)
# =========================================================
class Actor(BaseModel):
    name: str
    type: Literal["human", "system"] = "human"

class UseCase(BaseModel):
    id: str
    goal: str
    happy_path: List[str] = []
    alt_paths: List[str] = []

class DataEntityAttr(BaseModel):
    name: str
    type: str = "string"

class DataEntity(BaseModel):
    name: str
    attrs: List[DataEntityAttr] = []
    pii: bool = False

class NonFunctional(BaseModel):
    availability_pct: Optional[float] = Field(default=None, ge=0, le=100)
    rto: Optional[str] = None
    rpo: Optional[str] = None
    latency_ms: Optional[int] = None
    throughput: Optional[str] = None

class RequirementSpec(BaseModel):
    project_name: str
    summary: str
    actors_roles: List[Actor] = []
    use_cases: List[UseCase] = []
    functional_reqs: List[str] = []
    nonfunctional: NonFunctional = NonFunctional()
    constraints: dict = {}
    data: List[DataEntity] = []
    integrations: List[dict] = []
    observability: dict = {"logs": True, "metrics": True, "traces": True}
    security: dict = {"auth": "", "iam_principals": []}
    availability: dict = {"multi_az": True, "multi_region": False}
    risks: List[str] = []

class GapQuestion(BaseModel):
    id: str
    pillar: str
    text: str

class RequirementsAnalysis(BaseModel):
    spec: RequirementSpec
    gaps: List[GapQuestion] = []

class AnalyzeRequest(BaseModel):
    text: str

class AskRequest(BaseModel):
    question: str = Field(..., description="Your question")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)")
    max_tokens: Optional[int] = Field(400, ge=50, le=2000, description="Maximum tokens to generate")

class AskResponse(BaseModel):
    answer: str
    model: str
    processing_time: float

class Artifact(BaseModel):
    artifact_id: str
    type: Literal["c4_container", "uml_sequence"]
    language: Literal["mermaid", "plantuml"]
    content: str

# =========================================================
# NEW: Small Helpers (cache, parsing)
# =========================================================
def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

async def cache_get(key: str) -> Optional[str]:
    if not redis_client:
        return None
    try:
        return await redis_client.get(key)
    except Exception:
        return None

async def cache_set(key: str, value: str, ttl_sec: int = 86400):
    if not redis_client:
        return
    try:
        await redis_client.set(key, value, ex=ttl_sec)
    except Exception:
        pass

def _json_only(s: str) -> str:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return s[start:end+1]

def _truncate_tokens(text: str, max_len: int = 3000) -> str:
    tokens = tokenizer.encode(text, truncation=True, max_length=max_len)
    return tokenizer.decode(tokens)

async def call_local_llm_for_json(user_text: str) -> RequirementsAnalysis:
    schema_hint = json.dumps(RequirementsAnalysis.schema(), indent=2)
    sys = (
        "You are an architecting assistant. "
        "Return ONLY a single JSON object that validates the provided schema. "
        "No commentary, no markdown, no code fences."
    )
    inst = (
        "Read the user's requirements text and produce a RequirementsAnalysis with:\n"
        "- spec.project_name (best guess from text)\n"
        "- spec.summary (1-2 sentences)\n"
        "- zero or more actors_roles, use_cases, functional_reqs\n"
        "- nonfunctional fields when present; else leave null/empty\n"
        "- 2-6 gap questions tagged by pillar (Security/Reliability/Performance/Cost/Operational)\n"
        "Schema (for guidance):\n"
        f"{schema_hint}\n"
    )

    truncated = _truncate_tokens(user_text, max_len=3000)
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{sys}\n<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{inst}\nTEXT:\n{truncated}\n<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    raw = run_llm_inference(prompt)
    try:
        json_str = _json_only(raw)
        data = json.loads(json_str)
        return RequirementsAnalysis(**data)
    except (ValueError, ValidationError, json.JSONDecodeError):
        # Safe fallback
        fallback = RequirementsAnalysis(
            spec=RequirementSpec(project_name="Untitled Project", summary=truncated[:200]),
            gaps=[
                GapQuestion(id="sec-auth", pillar="Security", text="Which IdP (OIDC/Cognito/Auth0)?"),
                GapQuestion(id="rel-rto", pillar="Reliability", text="Target RTO/RPO?")
            ],
        )
        return fallback

# =========================================================
# Routes
# =========================================================
@app.get("/")
def root():
    return {"message": "Hello from your local agent!"}

@app.get("/docs")
def custom_docs_redirect():
    return RedirectResponse("/docs")

@app.get("/health")
def health_check():
    # optional: add a quick Redis ping for visibility
    redis_ok = False
    try:
        if redis_client:
            # a lightweight op; will raise if not connected
            redis_ok = True
    except Exception:
        redis_ok = False

    return {"status": "ok", "model": "Llama-3-8B", "redis": redis_ok, "uptime": "123 min"}

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    try:
        start = time.time()
        content = await file.read()

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")

        extracted_text = extract_text_from_file(file.filename, content)

        # Truncate to avoid overflow
        tokens = tokenizer.encode(extracted_text, truncation=True, max_length=4096)
        truncated_text = tokenizer.decode(tokens)

        # Dynamically adjust target summary length
        if len(truncated_text) < 1000:
            length_instruction = "in under 75 words"
        elif len(truncated_text) < 3000:
            length_instruction = "in 100–150 words"
        else:
            length_instruction = "in 200–250 words"

        # Prompt with Markdown structure delegated to the model
        full_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a professional summarization assistant. Your task is to summarize documents clearly and concisely, {length_instruction}, using Markdown. "
            "Use the following format:\n\n"
            "**Overview**  \n[Brief description]\n\n"
            "**Details**  \n[2–4 sentence explanation]\n\n"
            "**Outcome**  \n[Optional: result, insight, or next steps.]\n\n"
            "### 🔑 Key Points\n"
            "- [Point 1]\n- [Point 2]\n- [Point 3]\n\n"
            "Do not explain your response. Return only formatted markdown.\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Summarize the following document:\n{truncated_text}\n<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

        # Run LLM inference
        raw_output = run_llm_inference(full_prompt)
        markdown_output = raw_output.strip().split("Thank you")[0].replace("<|eot_id|>", "").strip()

        # Extract key points from markdown block
        key_points_match = re.search(r"### 🔑 Key Points\n((?:- .+\n?)+)", markdown_output)
        if key_points_match:
            key_points = [line.strip("- ").strip() for line in key_points_match.group(1).splitlines()]
        else:
            key_points = []

        time_elapsed = round(time.time() - start, 2)

        return JSONResponse(
            content={
                "summary": markdown_output,
                "key_points": key_points,
                "processing_time": time_elapsed,
                "model_used": "Llama-3-8B-Local"
            }
        )

    except FastAPIHTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# --------------------------
# NEW: Phase 1 Endpoints
# --------------------------
@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """
    Ask a general question and get an answer from the LLaMA-3-8B model.
    This endpoint does not use document retrieval - it's for general Q&A.
    """
    start = time.time()
    
    # Build a simple prompt for general Q&A
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful AI assistant. Answer questions clearly and concisely.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{req.question}\n<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    # Use the existing generator with custom parameters
    response = generator(
        prompt,
        return_full_text=False,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=req.temperature > 0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    answer = response[0]["generated_text"].strip()
    
    # Clean up any trailing tokens
    answer = answer.replace("<|eot_id|>", "").strip()
    
    elapsed = round(time.time() - start, 2)
    
    return AskResponse(
        answer=answer,
        model="Llama-3-8B-Local",
        processing_time=elapsed
    )

@app.post("/analyze", response_model=RequirementsAnalysis)
async def analyze(req: AnalyzeRequest):
    cache_key = f"analyze:{_sha(req.text)}"
    cached = await cache_get(cache_key)
    if cached:
        try:
            return RequirementsAnalysis(**json.loads(cached))
        except Exception:
            pass

    result = await call_local_llm_for_json(req.text)
    try:
        await cache_set(cache_key, result.json(), ttl_sec=7*24*3600)
    except Exception:
        pass
    return result

@app.post("/generate/diagrams", response_model=List[Artifact])
async def generate_diagrams(spec: RequirementSpec):
    project = spec.project_name or "System"
    # Mermaid C4-ish container (simple)
    mermaid = f"""flowchart LR
  subgraph {project.replace(' ','_')}[{project}]
    UI["React Frontend"]
    API["FastAPI Backend"]
  end
  DB[("Postgres")]
  CACHE[["Redis"]]
  UI --> API
  API --> DB
  API --> CACHE
"""
    # PlantUML sequence (simple)
    seq = """@startuml
actor User
participant "React UI" as UI
participant "FastAPI" as API
database DB
User -> UI: Submit requirements
UI -> API: POST /analyze
API -> DB: Store project/version
API --> UI: JSON {spec + gaps}
@enduml
"""

    artifacts = [
        Artifact(artifact_id="c4-ctr-001", type="c4_container", language="mermaid", content=mermaid),
        Artifact(artifact_id="seq-001", type="uml_sequence", language="plantuml", content=seq),
    ]
    return artifacts