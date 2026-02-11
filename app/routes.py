from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import cv2
import numpy as np
from app.processor import process_video_frames

router = APIRouter()

@router.post("/analyze")
async def analyze(
    frames: List[UploadFile] = File(...),
    scanId: str = Form(...)
):
    final_response = None

    for frame in frames:
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            continue

        response = process_video_frames(image, scanId)
        final_response = response

        # Stop early if scan is complete
        if response.get("status") == "success":
            break

    return final_response or {
        "status": "processing",
        "message": "Collecting frames"
    }
