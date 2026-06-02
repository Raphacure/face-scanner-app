"""Download document images (and PDFs) for OpenAI Vision."""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Tuple

import requests

MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
MAX_RENDERED_PAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 5
PDF_RENDER_DPI = 144
REQUEST_TIMEOUT_S = 30


def _download_bytes(url: str) -> bytes:
    headers = {"User-Agent": "face-ai-service/document-classify"}
    with requests.get(
        url,
        timeout=REQUEST_TIMEOUT_S,
        stream=True,
        headers=headers,
    ) as r:
        r.raise_for_status()
        data = bytearray()
        for chunk in r.iter_content(65536):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Document exceeds download size limit")
    raw = bytes(data)
    if not raw:
        raise ValueError("Empty document response")
    return raw


def _is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"


def _mime_from_image_bytes(data: bytes) -> str:
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and len(data) > 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if _is_pdf(data):
        return "application/pdf"
    raise ValueError(
        "Unsupported or invalid document format (use JPEG, PNG, GIF, WebP, or PDF)"
    )


def _pdf_to_png_pages(raw: bytes) -> List[bytes]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ValueError(
            "PDF support requires pymupdf; install with: pip install pymupdf"
        ) from e

    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")
        pages: List[bytes] = []
        for index in range(min(doc.page_count, MAX_PDF_PAGES)):
            pix = doc.load_page(index).get_pixmap(dpi=PDF_RENDER_DPI, alpha=False)
            png = pix.tobytes("png")
            if len(png) > MAX_RENDERED_PAGE_BYTES:
                raise ValueError(f"PDF page {index + 1} exceeds rendered size limit")
            pages.append(png)
        if doc.page_count > MAX_PDF_PAGES:
            # Vision still sees first MAX_PDF_PAGES pages; enough for typical claim PDFs.
            pass
        return pages
    finally:
        doc.close()


def _data_url_from_bytes(data: bytes, mime: str) -> str:
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def download_document_data_urls(url: str) -> Tuple[List[str], bytes]:
    """Fetch document and return one or more image data URLs for Vision."""
    raw = _download_bytes(url)
    mime = _mime_from_image_bytes(raw)

    if mime == "application/pdf":
        png_pages = _pdf_to_png_pages(raw)
        return [_data_url_from_bytes(page, "image/png") for page in png_pages], raw

    return [_data_url_from_bytes(raw, mime)], raw


def download_image_data_url(url: str) -> Tuple[str, bytes]:
    """Backward-compatible: first page/image data URL."""
    data_urls, raw = download_document_data_urls(url)
    return data_urls[0], raw


def build_vision_image_blocks(url: str) -> Tuple[List[Dict[str, Any]], bytes]:
    """OpenAI Vision content blocks (one block per PDF page or single image)."""
    data_urls, raw = download_document_data_urls(url)
    blocks = [
        {
            "type": "image_url",
            "image_url": {"url": data_url, "detail": "high"},
        }
        for data_url in data_urls
    ]
    return blocks, raw
