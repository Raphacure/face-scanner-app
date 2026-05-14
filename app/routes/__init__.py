import logging
import os
from typing import List, Optional

from fastapi import APIRouter, File, Form, Request, Response, UploadFile

from app.controllers.db_controller import db_health_check_controller
from app.controllers.scan_controller import analyze_controller
from app.controllers.user_controller import get_user_by_id_controller
from app.routes.db_routes import router as db_router
from app.routes.receipt_routes import router as receipt_router
from app.routes.scan_routes import router as scan_router
from app.routes.user_routes import router as user_router

router = APIRouter(prefix="/api/v1")
router.include_router(scan_router)
router.include_router(db_router)
router.include_router(receipt_router)
router.include_router(user_router)

logger = logging.getLogger(__name__)
USE_LEGACY_PATHS = os.getenv("USE_LEGACY_PATHS", "true").lower() == "true"

# TODO: Remove legacy aliases after client migration window closes.
if USE_LEGACY_PATHS:
    legacy_router = APIRouter(tags=["legacy"])

    def _set_deprecation_headers(response: Response, replacement_path: str) -> None:
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = os.getenv("LEGACY_PATHS_SUNSET", "2026-07-31")
        response.headers["Link"] = f'<{replacement_path}>; rel="successor-version"'

    @legacy_router.get("/db/health")
    def legacy_db_health(response: Response):
        logger.warning("Deprecated endpoint called: /db/health")
        _set_deprecation_headers(response, "/api/v1/db/health")
        return db_health_check_controller()

    @legacy_router.get("/users/{user_id}")
    def legacy_get_user(user_id: str, response: Response):
        logger.warning("Deprecated endpoint called: /users/{user_id}")
        _set_deprecation_headers(response, f"/api/v1/user/{user_id}")
        return get_user_by_id_controller(user_id)

    @legacy_router.post("/analyze")
    async def legacy_analyze(
        request: Request,
        response: Response,
        frames: List[UploadFile] = File(...),
        scanId: str = Form(...),
        userId: str = Form(...),
        clientId: Optional[str] = Form(None),
    ):
        logger.warning("Deprecated endpoint called: /analyze")
        _set_deprecation_headers(response, "/api/v1/facescan/analyze")
        return await analyze_controller(request, frames, scanId, userId, clientId)

    router.include_router(legacy_router)
