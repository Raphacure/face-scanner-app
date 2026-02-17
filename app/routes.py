from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import cv2
import numpy as np
import gc

from app.processor import process_video_frames

router = APIRouter()


@router.post("/analyze")
async def analyze(
    frames: List[UploadFile] = File(...),
    scanId: str = Form(...),
    userId: str = Form(...),
):
    final_response = None

    for frame in frames:
        try:
            contents = await frame.read()

            if not contents:
                continue

            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                continue

            response = process_video_frames(image, scanId, userId)
            final_response = response

            # ✅ Stop immediately if scan finished
            if response.get("status") == "success":
                break

        except Exception as e:
            # ✅ Never crash entire request because of one frame
            final_response = {
                "status": "error",
                "message": str(e)
            }

        finally:
            # ✅ CRITICAL for EC2 memory stability
            del contents
            del nparr
            del image
            gc.collect()

    return final_response or {
        "status": "processing",
        "message": "Collecting frames"
    }
