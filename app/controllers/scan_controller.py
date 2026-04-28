from typing import List, Optional
import gc

import cv2
import numpy as np
from fastapi import Request, UploadFile

from app.core.frame_buffer import clear
from app.processor import process_video_frames


async def analyze_controller(
    request: Request,
    frames: List[UploadFile],
    scan_id: str,
    user_id: str,
    client_id: Optional[str],
) -> dict:
    final_response = None

    for frame in frames:
        contents = None
        nparr = None
        image = None
        try:
            contents = await frame.read()

            if not contents:
                continue

            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                continue

            response = process_video_frames(request, image, scan_id, user_id, client_id)
            final_response = response

            if response.get("status") == "success":
                break

        except Exception as e:
            final_response = {"status": "error", "message": str(e)}

        finally:
            if contents is not None:
                del contents
            if nparr is not None:
                del nparr
            if image is not None:
                del image
            gc.collect()

    return final_response or {"status": "processing", "message": "Collecting frames"}


def cancel_scan_controller(payload: dict) -> dict:
    scan_id = payload.get("scanId")
    if not scan_id:
        return {"status": "error", "message": "scanId missing"}

    clear(scan_id)
    return {"status": "cancelled"}
