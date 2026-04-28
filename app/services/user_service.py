from typing import Optional

from app.models.user_model import User
from app.services.raw_sql_service import get_user_by_id as raw_get_user_by_id


def get_user_by_id(user_id: str) -> Optional[User]:
    result = raw_get_user_by_id(user_id)
    if not result["rows"]:
        return None
    return User.from_db_row(result["rows"][0])
