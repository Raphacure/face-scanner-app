from typing import Any, Dict, List

import os
import time

from app.core.openai_client import openai_configured
from app.services.openai_document_classifier import classify_document_url_openai

MAX_URLS = 25
_INTER_URL_DELAY_MS = max(0, int(os.getenv("OPENAI_INTER_URL_DELAY_MS", "500")))


def classify_receipt_prescription_urls_controller(urls: List[str]) -> Dict[str, Any]:
    if not openai_configured():
        return {
            "status": "error",
            "message": "OPENAI_API_KEY is required for document classification",
        }
    if not urls:
        return {"status": "error", "message": "urls must be a non-empty array"}
    if len(urls) > MAX_URLS:
        return {
            "status": "error",
            "message": f"At most {MAX_URLS} URLs allowed",
        }

    results: List[Dict[str, Any]] = []

    for index, url in enumerate(urls):
        if index > 0 and _INTER_URL_DELAY_MS > 0:
            time.sleep(_INTER_URL_DELAY_MS / 1000.0)
        u = (url or "").strip()
        if not u:
            results.append({"url": url or "", "error": "empty url"})
            continue
        try:
            results.append(classify_document_url_openai(u))
        except Exception as e:
            results.append({"url": u, "error": str(e)})

    return {"status": "success", "results": results}
