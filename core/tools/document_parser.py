"""Document parsing for all supported formats.

Features:
- Metadata extraction (PDF title, author, dates)
- Page-aware parsing (preserve page numbers)
- CSV handling (convert to markdown tables)
- SVG text extraction
- Image captioning via Florence-2 (with Tesseract OCR fallback)
"""

import io
import logging
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

log = logging.getLogger("noesis.parser")


class DocumentParser:
    """Parse documents and extract text + metadata."""

    def parse(self, filename: str, content: bytes) -> Dict:
        """Parse document and return structured data.

        Returns:
            {
                "text": str,              # Full document text
                "pages": List[dict],      # Page-by-page data (PDFs)
                "metadata": dict          # Title, author, etc.
            }
        """
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return self._parse_pdf(content)
        elif ext == ".docx":
            return self._parse_docx(content)
        elif ext in (".txt", ".md"):
            return self._parse_text(content)
        elif ext == ".csv":
            return self._parse_csv(content)
        elif ext in (".jpg", ".jpeg", ".png"):
            return self._parse_image(content)
        elif ext == ".svg":
            return self._parse_svg(content)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _parse_pdf(self, content: bytes) -> Dict:
        """Extract text + metadata from PDF with page preservation.

        Uses PyMuPDF (fitz) - already installed.
        """
        with fitz.open(stream=content, filetype="pdf") as pdf:
            metadata = pdf.metadata or {}
            pages = []

            for i, page in enumerate(pdf):
                page_text = page.get_text()
                pages.append({
                    "page_number": i + 1,
                    "text": page_text,
                })

            full_text = "\n\n".join(p["text"] for p in pages)

            return {
                "text": full_text,
                "pages": pages,
                "metadata": {
                    "title": metadata.get("title"),
                    "author": metadata.get("author"),
                    "created_date": metadata.get("creationDate"),
                    "page_count": len(pages),
                }
            }

    def _parse_docx(self, content: bytes) -> Dict:
        """Extract text from DOCX.

        Uses python-docx - already installed.
        """
        from docx import Document

        doc = Document(io.BytesIO(content))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n\n".join(paragraphs)

        return {
            "text": full_text,
            "metadata": {
                "paragraph_count": len(paragraphs),
            }
        }

    def _parse_text(self, content: bytes) -> Dict:
        """Extract text from plain text or markdown files."""
        text = content.decode(errors="ignore")

        return {
            "text": text.strip(),
            "metadata": {
                "encoding": "utf-8",
            }
        }

    def _parse_csv(self, content: bytes) -> Dict:
        """Parse CSV and convert to markdown table.

        Makes CSV data searchable in RAG.
        """
        import csv

        text_io = io.StringIO(content.decode(errors="ignore"))
        reader = csv.reader(text_io)

        rows = list(reader)
        if not rows:
            return {"text": "", "metadata": {}}

        # Convert to markdown table
        header = rows[0]
        markdown_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]

        for row in rows[1:]:
            if len(row) == len(header):  # Skip malformed rows
                markdown_lines.append("| " + " | ".join(row) + " |")

        markdown_text = "\n".join(markdown_lines)

        return {
            "text": markdown_text,
            "metadata": {
                "row_count": len(rows),
                "column_count": len(header),
            }
        }

    def _parse_image(self, content: bytes) -> Dict:
        """Extract caption + OCR text from image.

        Uses Florence-2 for semantic captioning and OCR.
        Falls back to Tesseract if Florence-2 is unavailable.
        """
        image = Image.open(io.BytesIO(content))
        width, height = image.width, image.height
        fmt = image.format

        try:
            from core.vision.captioner import caption_and_ocr
            result = caption_and_ocr(content)
            caption = result["caption"]
            ocr_text = result["ocr_text"]
            # Combine caption + OCR for the main text field (used for embedding)
            parts = []
            if caption:
                parts.append(caption)
            if ocr_text:
                parts.append(ocr_text)
            text = "\n\n".join(parts) if parts else "[Image with no extractable content]"

            return {
                "text": text,
                "image_caption": caption,
                "ocr_text": ocr_text,
                "metadata": {
                    "vision_model": "florence-2",
                    "width": width,
                    "height": height,
                    "format": fmt,
                }
            }
        except Exception as e:
            log.warning("Florence-2 unavailable, falling back to Tesseract: %s", e)
            ocr_text = pytesseract.image_to_string(image, lang="eng").strip()
            return {
                "text": ocr_text if ocr_text else "[Image with no extractable content]",
                "image_caption": "",
                "ocr_text": ocr_text,
                "metadata": {
                    "ocr": True,
                    "vision_model": "tesseract-fallback",
                    "width": width,
                    "height": height,
                    "format": fmt,
                }
            }

    def _parse_svg(self, content: bytes) -> Dict:
        """Extract <text> elements from SVG diagrams.

        SVG files often contain diagrams with text labels.
        """
        try:
            tree = ET.fromstring(content)
            texts = []

            # Find all <text> elements (SVG text nodes)
            # SVG namespace: {http://www.w3.org/2000/svg}
            for elem in tree.iter():
                # Check for text elements (with or without namespace)
                if elem.tag.endswith("text") or elem.tag == "text":
                    if elem.text and elem.text.strip():
                        texts.append(elem.text.strip())

            full_text = "\n".join(texts)

            return {
                "text": full_text,
                "metadata": {
                    "svg_elements": len(list(tree.iter())),
                    "text_nodes": len(texts),
                }
            }
        except ET.ParseError as e:
            return {
                "text": "",
                "metadata": {
                    "error": f"SVG parse error: {str(e)}",
                }
            }
