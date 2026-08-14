"""File-backed store for async classify-receipt jobs.

Jobs are written to disk so poll requests work across uvicorn workers
and brief process restarts on the same host.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

_lock = Lock()

JOB_TTL_S = max(300, int(os.getenv("CLASSIFY_JOB_TTL_S", "3600")))
_JOB_DIR = Path(os.getenv("CLASSIFY_JOB_DIR") or "/tmp/face-ai-classify-jobs")


def _job_path(job_id: str) -> Path:
    return _JOB_DIR / f"{job_id}.json"


def _ensure_dir() -> None:
    _JOB_DIR.mkdir(parents=True, exist_ok=True)


def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    created_at = float(data.get("created_at") or 0)
    if created_at and time.time() - created_at > JOB_TTL_S:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return data


def _write_job(job: Dict[str, Any]) -> None:
    _ensure_dir()
    job_id = str(job.get("job_id") or "")
    if not job_id:
        return
    path = _job_path(job_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job), encoding="utf-8")
    tmp.replace(path)


def _cleanup_expired_locked() -> None:
    _ensure_dir()
    now = time.time()
    for path in _JOB_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            created_at = float((data or {}).get("created_at") or 0)
            if not created_at or now - created_at > JOB_TTL_S:
                path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _cleanup_expired_locked()
        _write_job(
            {
                "job_id": job_id,
                "status": "pending",
                "created_at": time.time(),
                "completed_at": None,
                "result": None,
                "error": None,
            }
        )
    return job_id


def set_job_processing(job_id: str) -> None:
    with _lock:
        job = _read_job(job_id)
        if job:
            job["status"] = "processing"
            _write_job(job)


def set_job_completed(job_id: str, result: Dict[str, Any]) -> None:
    with _lock:
        job = _read_job(job_id)
        if job:
            job["status"] = "completed"
            job["completed_at"] = time.time()
            job["result"] = result
            job["error"] = None
            _write_job(job)


def set_job_failed(job_id: str, error: str) -> None:
    with _lock:
        job = _read_job(job_id)
        if job:
            job["status"] = "failed"
            job["completed_at"] = time.time()
            job["error"] = error
            job["result"] = None
            _write_job(job)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _read_job(job_id)
        if not job:
            return None
        return dict(job)
