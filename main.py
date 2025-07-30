import io
import re
import time
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from docx import Document
import torch
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# --- FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.relufox.ai",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model ---
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

# --- LLM Inference ---
def run_llm_inference(prompt: str) -> str:
    response = generator(prompt, return_full_text=False)
    return response[0]["generated_text"].strip()

def extract_text_from_file(filename: str, content: bytes) -> str:
    ext = filename.lower()
    text = ""

    if ext.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])

    elif ext.endswith(".pdf"):
        with fitz.open(stream=content, filetype="pdf") as pdf:
            text = "\n".join([page.get_text() for page in pdf])

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

# --- Routes ---
@app.get("/")
def root():
    return {"message": "Hello from your local agent!"}

@app.get("/docs")
def custom_docs_redirect():
    return RedirectResponse("/docs")

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "Llama-3-8B", "uptime": "123 min"}

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
            },
            headers={"Access-Control-Allow-Origin": "https://app.relufox.ai"}
        )

    except FastAPIHTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")