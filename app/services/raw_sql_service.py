import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
import json

import psycopg2
from psycopg2.extras import RealDictCursor


def _database_dsn() -> str:
    required_keys = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASS", "DB_PORT"]
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing DB environment variables: {', '.join(missing)}")

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    return f"host={host} port={port} dbname={db_name} user={user} password={password}"


@contextmanager
def get_conn() -> Generator["psycopg2.extensions.connection", None, None]:
    conn = psycopg2.connect(_database_dsn())
    try:
        yield conn
    finally:
        conn.close()


def _result(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"rows": rows, "rowcount": len(rows)}


def health_check() -> Dict[str, Any]:
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT 1 AS status")
        rows = cur.fetchall() or []
        return _result([dict(r) for r in rows])


def get_user_details_by_id(user_id: str) -> Dict[str, Any]:
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT first_name, last_name, age, gender, email, phone
            FROM users
            WHERE id = %s
            LIMIT 1
            """,
            (user_id,),
        )
        row = cur.fetchone()
        return _result([dict(row)] if row else [])


def get_user_by_id(user_id: str) -> Dict[str, Any]:
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, first_name, last_name, age, gender, email, phone
            FROM users
            WHERE id = %s
            LIMIT 1
            """,
            (user_id,),
        )
        row = cur.fetchone()
        return _result([dict(row)] if row else [])


def insert_face_scan_record(scan_data: Dict[str, Any]) -> None:
    device = scan_data.get("device")
    ip = scan_data.get("ip")
    user_id = scan_data.get("user_id")
    client_id = scan_data.get("client_id")
    report_url = scan_data.get("report_url") or []
    image_url = scan_data.get("image_url")
    response = scan_data.get("response") or {}

    # Ensure JSON is saved properly even if callers pass non-JSON-serializable objects.
    response_json: Optional[str]
    try:
        response_json = json.dumps(response)
    except TypeError:
        response_json = json.dumps({"raw": str(response)})

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO face_scan (device, ip, user_id, client_id, report_url, image_url, response)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (device, ip, user_id, client_id, report_url, image_url, response_json),
        )
        conn.commit()

