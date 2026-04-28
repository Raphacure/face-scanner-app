from typing import Any, Dict

from app.db import run_query


def get_user_details_by_id(user_id: str) -> Dict[str, Any]:
    return run_query(
        """
        SELECT first_name, last_name, age, gender, email, phone
        FROM users
        WHERE id = %s
        LIMIT 1
        """,
        (user_id,),
    )


def get_user_by_id(user_id: str) -> Dict[str, Any]:
    return run_query(
        """
        SELECT id, first_name, last_name, age, gender, email, phone
        FROM users
        WHERE id = %s
        LIMIT 1
        """,
        (user_id,),
    )

