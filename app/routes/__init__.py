from fastapi import APIRouter

from app.routes.db_routes import router as db_router
from app.routes.scan_routes import router as scan_router
from app.routes.user_routes import router as user_router

router = APIRouter()
router.include_router(scan_router)
router.include_router(db_router)
router.include_router(user_router)
