from app.services.user_service import get_user_by_id


def get_user_by_id_controller(user_id: str) -> dict:
    try:
        user = get_user_by_id(user_id)
        if user is None:
            return {"status": "error", "message": "User not found"}
        return {"status": "success", "user": user.to_dict()}
    except Exception as e:
        return {"status": "error", "message": str(e)}
