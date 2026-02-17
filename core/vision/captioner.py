"""Image captioning via Florence-2 (Microsoft, MIT license).

Runs on CPU to avoid competing with vLLM for GPU memory.
Model: microsoft/Florence-2-large (770M params, ~3GB RAM).

Provides:
  - Detailed captions for uploaded images
  - OCR text extraction (Florence-2 has built-in OCR)
  - On-demand captioning via get_caption()
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache

import torch
from PIL import Image

from core.config import settings

log = logging.getLogger("noesis.vision")


@lru_cache(maxsize=1)
def _load_model():
    """Load Florence-2 model and processor once, cached for reuse.

    Forces CPU to avoid competing with vLLM for GPU memory.
    """
    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(
        settings.noesis_vision_model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.noesis_vision_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    log.info("Loaded %s on CPU", settings.noesis_vision_model)
    return processor, model


def _run_task(image: Image.Image, task: str, text_input: str = "") -> str:
    """Run a Florence-2 task on an image."""
    processor, model = _load_model()

    prompt = task if not text_input else task + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
        )

    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        result, task=task, image_size=(image.width, image.height)
    )
    return parsed.get(task, "")


def get_caption(image_bytes: bytes) -> str:
    """Generate a detailed caption for an image.

    Returns a natural language description of the image content.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _run_task(image, "<MORE_DETAILED_CAPTION>")


def get_ocr_text(image_bytes: bytes) -> str:
    """Extract text from an image using Florence-2's built-in OCR.

    More context-aware than Tesseract for mixed text/visual content.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _run_task(image, "<OCR>")


def caption_and_ocr(image_bytes: bytes) -> dict:
    """Get both caption and OCR text in one call.

    Returns:
        {"caption": str, "ocr_text": str}
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    caption = _run_task(image, "<MORE_DETAILED_CAPTION>")
    ocr_text = _run_task(image, "<OCR>")
    return {"caption": caption, "ocr_text": ocr_text}
