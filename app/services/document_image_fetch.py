"""Download document images (and PDFs) for OpenAI Vision."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import requests

MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
MAX_RENDERED_PAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 5
REQUEST_TIMEOUT_S = 60
_VALID_IMAGE_DETAIL = frozenset({"auto", "low", "high"})


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(lo, min(hi, value))


def _pdf_render_dpi() -> int:
    return _env_int("PDF_RENDER_DPI", 120, 72, 200)


def pdf_vision_max_pages() -> int:
    """Pages sent on the main classification call (full PDF can have many pages)."""
    return _env_int("PDF_VISION_MAX_PAGES", 1, 1, MAX_PDF_PAGES)


def pdf_refine_max_pages() -> int:
    """Pages sent on refine/regulatory passes (bill header usually on page 1)."""
    return _env_int("PDF_REFINE_MAX_PAGES", 1, 1, MAX_PDF_PAGES)


def _vision_image_detail() -> str:
    detail = (os.getenv("OPENAI_IMAGE_DETAIL") or "auto").strip().lower()
    return detail if detail in _VALID_IMAGE_DETAIL else "auto"


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


def is_pdf_bytes(data: bytes) -> bool:
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
    if is_pdf_bytes(data):
        return "application/pdf"
    raise ValueError(
        "Unsupported or invalid document format (use JPEG, PNG, GIF, WebP, or PDF)"
    )


def _pixmap_to_jpeg(pix: Any) -> bytes:
    try:
        jpeg = pix.tobytes("jpeg", jpg_quality=85)
        if jpeg:
            return jpeg
    except Exception:
        pass
    png = pix.tobytes("png")
    if len(png) > MAX_RENDERED_PAGE_BYTES:
        raise ValueError("PDF page exceeds rendered size limit")
    return png


def render_pdf_page_images(raw: bytes, max_pages: int) -> List[bytes]:
    """Render PDF pages to JPEG (or PNG fallback) for Vision."""
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
        limit = min(doc.page_count, max(1, max_pages))
        pages: List[bytes] = []
        dpi = _pdf_render_dpi()
        for index in range(limit):
            pix = doc.load_page(index).get_pixmap(dpi=dpi, alpha=False)
            page_bytes = _pixmap_to_jpeg(pix)
            if len(page_bytes) > MAX_RENDERED_PAGE_BYTES:
                raise ValueError(f"PDF page {index + 1} exceeds rendered size limit")
            pages.append(page_bytes)
        return pages
    finally:
        doc.close()


def pdf_page_count(raw: bytes) -> int:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return 0
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


def _page_mime(page_bytes: bytes) -> str:
    if page_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/png"


def _data_url_from_bytes(data: bytes, mime: str) -> str:
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


@dataclass
class DocumentPages:
    """Downloaded document with page images cached (PDF rendered once per request)."""

    raw: bytes
    page_images: List[bytes]

    @property
    def is_pdf(self) -> bool:
        return is_pdf_bytes(self.raw)


def load_document(url: str) -> DocumentPages:
    raw = _download_bytes(url)
    if is_pdf_bytes(raw):
        max_render = max(pdf_vision_max_pages(), pdf_refine_max_pages())
        return DocumentPages(raw=raw, page_images=render_pdf_page_images(raw, max_render))
    return DocumentPages(raw=raw, page_images=[raw])


def _blocks_from_page_bytes(
    pages: List[bytes],
    detail: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_from_bytes(page, _page_mime(page)),
                "detail": detail,
            },
        }
        for page in pages
    ]


def download_document_data_urls(url: str) -> Tuple[List[str], bytes]:
    """Fetch document and return one or more image data URLs for Vision."""
    raw = _download_bytes(url)
    mime = _mime_from_image_bytes(raw)

    if mime == "application/pdf":
        pages = render_pdf_page_images(raw, pdf_vision_max_pages())
        return [_data_url_from_bytes(p, _page_mime(p)) for p in pages], raw

    return [_data_url_from_bytes(raw, mime)], raw


def download_image_data_url(url: str) -> Tuple[str, bytes]:
    """Backward-compatible: first page/image data URL."""
    data_urls, raw = download_document_data_urls(url)
    return data_urls[0], raw


def build_vision_image_blocks(url: str) -> Tuple[List[Dict[str, Any]], bytes]:
    """OpenAI Vision blocks — PDFs limited to PDF_VISION_MAX_PAGES (default 1)."""
    doc = load_document(url)
    detail = _vision_image_detail()
    limit = pdf_vision_max_pages() if doc.is_pdf else len(doc.page_images)
    return _blocks_from_page_bytes(doc.page_images[:limit], detail), doc.raw


def build_vision_blocks_from_document(
    doc: DocumentPages,
) -> List[Dict[str, Any]]:
    detail = _vision_image_detail()
    limit = pdf_vision_max_pages() if doc.is_pdf else len(doc.page_images)
    return _blocks_from_page_bytes(doc.page_images[:limit], detail)


def build_pdf_refine_blocks(
    document_raw: bytes,
    detail: str = "high",
    doc: DocumentPages | None = None,
) -> List[Dict[str, Any]]:
    """Fewer PDF pages for refine passes (default 1) to avoid timeouts."""
    if doc is not None and doc.is_pdf:
        limit = pdf_refine_max_pages()
        return _blocks_from_page_bytes(doc.page_images[:limit], detail)
    if not is_pdf_bytes(document_raw):
        return []
    pages = render_pdf_page_images(document_raw, pdf_refine_max_pages())
    return _blocks_from_page_bytes(pages, detail)


def _header_crop_top_ratio() -> float:
    try:
        ratio = float(os.getenv("PHARMACY_HEADER_CROP_RATIO", "0.38"))
    except ValueError:
        ratio = 0.38
    return max(0.15, min(0.55, ratio))


def build_header_crop_data_url(image_bytes: bytes) -> str | None:
    """Crop top band of a bill image — GST/DL are often printed in this margin."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    height = img.shape[0]
    crop_height = max(1, int(height * _header_crop_top_ratio()))
    top_band = img[0:crop_height, :]
    ok, encoded = cv2.imencode(".jpg", top_band, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    return _data_url_from_bytes(encoded.tobytes(), "image/jpeg")


def build_header_crop_from_document(
    document_raw: bytes,
    doc: DocumentPages | None = None,
) -> str | None:
    """Header crop for images or PDF (uses first rendered page)."""
    if doc is not None and doc.page_images:
        return build_header_crop_data_url(doc.page_images[0])
    if is_pdf_bytes(document_raw):
        pages = render_pdf_page_images(document_raw, 1)
        if not pages:
            return None
        return build_header_crop_data_url(pages[0])
    return build_header_crop_data_url(document_raw)


def build_regulatory_header_blocks(
    image_blocks: List[Dict[str, Any]],
    document_raw: bytes,
    doc: DocumentPages | None = None,
) -> List[Dict[str, Any]]:
    """Refine/regulatory vision blocks — PDFs use fewer pages + header crop."""
    if is_pdf_bytes(document_raw):
        blocks = build_pdf_refine_blocks(document_raw, detail="high", doc=doc)
    else:
        blocks = []
        for block in image_blocks:
            if block.get("type") != "image_url":
                blocks.append(block)
                continue
            image_url = block.get("image_url")
            if not isinstance(image_url, dict):
                blocks.append(block)
                continue
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {**image_url, "detail": "high"},
                }
            )

    crop_url = build_header_crop_from_document(document_raw, doc=doc)
    if crop_url:
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": crop_url, "detail": "high"},
            }
        )
    return blocks


def build_refine_image_blocks(
    image_blocks: List[Dict[str, Any]],
    document_raw: bytes,
    detail: str = "high",
    doc: DocumentPages | None = None,
) -> List[Dict[str, Any]]:
    """Invoice/Rx/report refine blocks — limit PDF pages to avoid multi-minute requests."""
    pdf_blocks = build_pdf_refine_blocks(document_raw, detail=detail, doc=doc)
    if pdf_blocks:
        return pdf_blocks

    blocks: List[Dict[str, Any]] = []
    for block in image_blocks:
        if block.get("type") != "image_url":
            blocks.append(block)
            continue
        image_url = block.get("image_url")
        if not isinstance(image_url, dict):
            blocks.append(block)
            continue
        blocks.append(
            {
                "type": "image_url",
                "image_url": {**image_url, "detail": detail},
            }
        )
    return blocks
