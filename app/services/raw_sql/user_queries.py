from typing import Any, Dict

from app.db import run_query

_USER_FIELDS = "first_name, last_name, age, gender, email, phone"


def _get_user_by_id_with_cols(columns: str, user_id: str) -> Dict[str, Any]:
    return run_query(
        f"""
        SELECT {columns}
        FROM users
        WHERE id = %s
        LIMIT 1
        """,
        (user_id,),
    )


def get_user_details_by_id(user_id: str) -> Dict[str, Any]:
    return _get_user_by_id_with_cols(_USER_FIELDS, user_id)


def get_user_by_id(user_id: str) -> Dict[str, Any]:
    return _get_user_by_id_with_cols(f"id, {_USER_FIELDS}", user_id)

