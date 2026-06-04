from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import os
import time

from app.core.openai_client import openai_configured
from app.services.openai_document_classifier import classify_document_url_openai

MAX_URLS = 25
_INTER_URL_DELAY_MS = max(0, int(os.getenv("OPENAI_INTER_URL_DELAY_MS", "0")))


def _classify_parallel_workers(url_count: int) -> int:
    try:
        workers = int(os.getenv("OPENAI_CLASSIFY_MAX_PARALLEL", "2"))
    except ValueError:
        workers = 2
    return max(1, min(workers, url_count, MAX_URLS))


def _classify_one(index: int, url: str) -> Tuple[int, Dict[str, Any]]:
    if index > 0 and _INTER_URL_DELAY_MS > 0:
        time.sleep(_INTER_URL_DELAY_MS / 1000.0)
    u = (url or "").strip()
    if not u:
        return index, {"url": url or "", "error": "empty url"}
    try:
        return index, classify_document_url_openai(u)
    except Exception as e:
        return index, {"url": u, "error": str(e)}


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

    workers = _classify_parallel_workers(len(urls))
    indexed_results: List[Tuple[int, Dict[str, Any]]] = []

    if workers <= 1:
        for index, url in enumerate(urls):
            indexed_results.append(_classify_one(index, url))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_classify_one, index, url) for index, url in enumerate(urls)
            ]
            for future in as_completed(futures):
                indexed_results.append(future.result())

    indexed_results.sort(key=lambda item: item[0])
    results = [item[1] for item in indexed_results]
    return {"status": "success", "results": results}
