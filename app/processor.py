import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
import gc

from app.core.frame_buffer import add_frame, is_ready, get_frames, clear, count
from app.core.aggregator import calculate_all
from app.quality.quality_aggregator import evaluate_scan_quality

# ✅ Create ONCE (GLOBAL / PER WORKER)
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


def process_video_frames(frame, scan_id):

    small = cv2.resize(frame, (320, 240))   # ✅ Huge memory reduction
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    results = face_detector.process(rgb)

    if not results.detections:
        return {
            "status": "error",
            "message": "No face detected"
        }

    add_frame(scan_id, rgb)

    if not is_ready(scan_id):
        return {
            "status": "processing",
            "message": f"Collecting frames {count(scan_id)}/300"
        }

    frames = get_frames(scan_id)

    quality = evaluate_scan_quality(frames)
    data = calculate_all(frames)

    clear(scan_id)

    # ✅ Critical cleanup
    del frames
    del rgb
    del small
    gc.collect()

    return {
        "status": "success",
        "quality": quality,
        "data": data
    }
