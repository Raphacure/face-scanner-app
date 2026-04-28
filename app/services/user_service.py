from typing import Optional

from app.models.user_model import User
from app.services.database_service import execute_query


def get_user_by_id(user_id: str) -> Optional[User]:
    result = execute_query("SELECT * FROM users WHERE id = %s LIMIT 1", (user_id,))
    if not result["rows"]:
        return None
    return User.from_db_row(result["rows"][0])
