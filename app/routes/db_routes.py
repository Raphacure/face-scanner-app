from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.controllers.db_controller import db_health_check_controller, db_query_controller

router = APIRouter(prefix="/db", tags=["db"])


class DBReadRequest(BaseModel):
    query_name: str


@router.get("/health")
def db_health_check():
    return db_health_check_controller()


@router.post("/read")
def db_read(payload: DBReadRequest, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return db_query_controller(payload.query_name)
