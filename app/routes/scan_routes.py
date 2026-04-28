from fastapi import APIRouter, File, Form, Request, UploadFile
from typing import List, Optional

from app.controllers.scan_controller import analyze_controller, cancel_scan_controller

router = APIRouter(prefix="/facescan", tags=["facescan"])


@router.post("/analyze")
async def analyze(
    request: Request,
    frames: List[UploadFile] = File(...),
    scanId: str = Form(...),
    userId: str = Form(...),
    clientId: Optional[str] = Form(None),
):
    return await analyze_controller(request, frames, scanId, userId, clientId)


@router.post("/cancel-scan")
def cancel_scan(payload: dict):
    return cancel_scan_controller(payload)
