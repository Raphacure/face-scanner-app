from fastapi import APIRouter

from app.controllers.user_controller import get_user_by_id_controller

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/{user_id}")
def get_user_by_id(user_id: str):
    return get_user_by_id_controller(user_id)
