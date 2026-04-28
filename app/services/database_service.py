import os
from typing import Any, Dict, Tuple


def _connection_kwargs() -> Dict[str, Any]:
    return {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS"),
        "connect_timeout": 5,
    }


def _ensure_db_config() -> None:
    required_keys = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASS", "DB_PORT"]
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing DB environment variables: {', '.join(missing)}")


def execute_query(query: str, params: Tuple[Any, ...] = ()) -> Dict[str, Any]:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL driver missing. Install with: pip install psycopg2-binary"
        ) from exc

    _ensure_db_config()

    with psycopg2.connect(**_connection_kwargs()) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            if cursor.description:
                rows = cursor.fetchall()
                return {"rows": [dict(row) for row in rows], "rowcount": len(rows)}
            return {"rows": [], "rowcount": cursor.rowcount}
