from typing import Any, Dict

from app.db import run_query


def health_check() -> Dict[str, Any]:
    return run_query("SELECT 1 AS status")

