from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import logging
import os
import time

from app.core.openai_client import openai_configured
from app.services.classify_job_store import (
    create_job,
    get_job,
    set_job_completed,
    set_job_failed,
    set_job_processing,
)
from app.services.openai_document_classifier import classify_document_url_openai

logger = logging.getLogger(__name__)

MAX_URLS = 25
_INTER_URL_DELAY_MS = max(0, int(os.getenv("OPENAI_INTER_URL_DELAY_MS", "0")))
_JOB_POLL_INTERVAL_MS = max(500, int(os.getenv("CLASSIFY_JOB_POLL_INTERVAL_MS", "2000")))

_job_executor = ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("CLASSIFY_JOB_MAX_PARALLEL", "3")))
)


def _classify_parallel_workers(url_count: int) -> int:
    try:
        workers = int(os.getenv("OPENAI_CLASSIFY_MAX_PARALLEL", "6"))
    except ValueError:
        workers = 6
    return max(1, min(workers, url_count, MAX_URLS))


def _normalize_items(urls: List[Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for entry in urls:
        if isinstance(entry, dict):
            item: Dict[str, Any] = {
                "url": str(entry.get("url") or "").strip(),
                "name": str(entry.get("name") or "").strip(),
            }
            fields = entry.get("fields")
            if fields:
                item["fields"] = list(fields)
        else:
            item = {
                "url": str(getattr(entry, "url", "") or "").strip(),
                "name": str(getattr(entry, "name", "") or "").strip(),
            }
            fields = getattr(entry, "fields", None)
            if fields:
                item["fields"] = list(fields)
        items.append(item)
    return items


def _classify_one(index: int, url: str, name: str, fields: List[str] | None = None) -> Tuple[int, Dict[str, Any]]:
    if index > 0 and _INTER_URL_DELAY_MS > 0:
        time.sleep(_INTER_URL_DELAY_MS / 1000.0)
    u = (url or "").strip()
    if not u:
        return index, {"url": url or "", "name": name, "error": "empty url"}
    try:
        return index, classify_document_url_openai(
            u, category_hint=name, extract_fields=fields
        )
    except Exception as e:
        return index, {"url": u, "name": name, "error": str(e)}


def _run_classify_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_started = time.perf_counter()
    workers = _classify_parallel_workers(len(items))
    indexed_results: List[Tuple[int, Dict[str, Any]]] = []

    if workers <= 1:
        for index, item in enumerate(items):
            indexed_results.append(
                _classify_one(index, item["url"], item["name"], item.get("fields"))
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _classify_one,
                    index,
                    item["url"],
                    item["name"],
                    item.get("fields"),
                )
                for index, item in enumerate(items)
            ]
            for future in as_completed(futures):
                indexed_results.append(future.result())

    indexed_results.sort(key=lambda item: item[0])
    results = [item[1] for item in indexed_results]
    return {
        "status": "success",
        "mode": "hybrid",
        "processing_time_ms": round((time.perf_counter() - batch_started) * 1000.0, 1),
        "results": results,
    }


def _validate_classify_request(urls: List[Any]) -> Dict[str, Any] | None:
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
    return None


def classify_receipt_prescription_urls_controller(urls: List[Any]) -> Dict[str, Any]:
    error = _validate_classify_request(urls)
    if error:
        return error
    return _run_classify_batch(_normalize_items(urls))


def _process_classify_job(job_id: str, items: List[Dict[str, Any]]) -> None:
    set_job_processing(job_id)
    try:
        result = _run_classify_batch(items)
        if result.get("status") == "error":
            set_job_failed(job_id, str(result.get("message") or "classification failed"))
        else:
            set_job_completed(job_id, result)
    except Exception as exc:
        logger.exception("async classify job %s failed", job_id)
        set_job_failed(job_id, str(exc))


def submit_classify_job_controller(urls: List[Any]) -> Dict[str, Any]:
    error = _validate_classify_request(urls)
    if error:
        return error

    items = _normalize_items(urls)
    job_id = create_job()
    _job_executor.submit(_process_classify_job, job_id, items)
    return {
        "status": "accepted",
        "job_id": job_id,
        "poll_url": f"/api/v1/documents/classify-receipt/jobs/{job_id}",
        "poll_interval_ms": _JOB_POLL_INTERVAL_MS,
        "message": (
            "Classification started. Poll poll_url every poll_interval_ms until "
            "status is success or error."
        ),
    }


def get_classify_job_controller(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job:
        return {"status": "error", "message": "job not found or expired", "job_id": job_id}

    state = str(job.get("status") or "")
    if state in ("pending", "processing"):
        return {
            "status": state,
            "job_id": job_id,
            "message": "Classification in progress",
        }

    if state == "failed":
        return {
            "status": "error",
            "job_id": job_id,
            "message": str(job.get("error") or "classification failed"),
        }

    result = job.get("result")
    if isinstance(result, dict):
        return {"job_id": job_id, **result}
    return {
        "status": "error",
        "job_id": job_id,
        "message": "job completed without result",
    }
