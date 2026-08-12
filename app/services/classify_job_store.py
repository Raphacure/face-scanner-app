"""In-memory store for async classify-receipt jobs (single-instance friendly)."""

from __future__ import annotations

import os
import time
import uuid
from threading import Lock
from typing import Any, Dict, Optional

_lock = Lock()
_jobs: Dict[str, Dict[str, Any]] = {}

JOB_TTL_S = max(300, int(os.getenv("CLASSIFY_JOB_TTL_S", "3600")))


def _cleanup_expired_locked() -> None:
    now = time.time()
    expired = [
        job_id
        for job_id, job in _jobs.items()
        if now - float(job.get("created_at", now)) > JOB_TTL_S
    ]
    for job_id in expired:
        _jobs.pop(job_id, None)


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _cleanup_expired_locked()
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": time.time(),
            "completed_at": None,
            "result": None,
            "error": None,
        }
    return job_id


def set_job_processing(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job["status"] = "processing"


def set_job_completed(job_id: str, result: Dict[str, Any]) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job["status"] = "completed"
            job["completed_at"] = time.time()
            job["result"] = result
            job["error"] = None


def set_job_failed(job_id: str, error: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job["status"] = "failed"
            job["completed_at"] = time.time()
            job["error"] = error
            job["result"] = None


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        _cleanup_expired_locked()
        job = _jobs.get(job_id)
        if not job:
            return None
        return dict(job)
