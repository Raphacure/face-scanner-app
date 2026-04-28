"""
Backward-compatible imports for legacy call sites.
Prefer `app.services.raw_sql.*_queries` modules for new code.
"""

from app.services.raw_sql.face_scan_queries import insert_face_scan_record
from app.services.raw_sql.health_queries import health_check
from app.services.raw_sql.user_queries import get_user_by_id, get_user_details_by_id

__all__ = [
    "health_check",
    "get_user_details_by_id",
    "get_user_by_id",
    "insert_face_scan_record",
]

