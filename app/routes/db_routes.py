from fastapi import APIRouter

from app.controllers.db_controller import db_health_check_controller, db_query_controller

router = APIRouter(prefix="/db", tags=["db"])


@router.get("/health")
def db_health_check():
    return db_health_check_controller()


@router.get("/read")
def db_read(query: str):
    return db_query_controller(query)
