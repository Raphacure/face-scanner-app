from typing import Any, Dict, Tuple

from app.services.database_service import execute_query


def run_query(query: str, params: Tuple[Any, ...] = ()) -> Dict[str, Any]:
    return execute_query(query, params)


def run_read_query(query: str, params: Tuple[Any, ...] = ()) -> Dict[str, Any]:
    # Backward-compatible wrapper around the new service layer.
    return run_query(query, params)
