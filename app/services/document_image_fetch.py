"""Download document images for OpenAI (no OpenCV)."""

from __future__ import annotations

import base64
from typing import Tuple

import requests

MAX_IMAGE_BYTES = 10 * 1024 * 1024
REQUEST_TIMEOUT_S = 20


def _mime_from_bytes(data: bytes) -> str:
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and len(data) > 12 and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported or invalid image format")


def download_image_data_url(url: str) -> Tuple[str, bytes]:
    """Fetch image bytes and return (data URL, raw bytes)."""
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
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds size limit")
    raw = bytes(data)
    if not raw:
        raise ValueError("Empty image response")
    mime = _mime_from_bytes(raw)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", raw
