from app.core.frame_buffer import clear
from fastapi import APIRouter, UploadFile, File, Form,Request
from typing import List
import cv2
import numpy as np
import gc

from app.processor import process_video_frames

router = APIRouter()


@router.post("/analyze")
async def analyze(
    request: Request,
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

            response = process_video_frames(request,image, scanId, userId)
            final_response = response
            # todo

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

@router.post("/cancel-scan")
def cancel_scan(payload: dict):
    scan_id = payload.get("scanId")

    if not scan_id:
        return {"status": "error", "message": "scanId missing"}

    clear(scan_id)

    return {"status": "cancelled"}
