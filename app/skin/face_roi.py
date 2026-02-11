import numpy as np
from app.core.landmarks import get_face_landmarks

# Simple cheek-based ROI for skin analysis
CHEEK_POINTS = [50, 187, 205, 36, 118, 119]

def extract_face_roi(frame):
    landmarks = get_face_landmarks(frame)
    if not landmarks:
        return None

    h, w, _ = frame.shape
    xs = [int(landmarks.landmark[i].x * w) for i in CHEEK_POINTS]
    ys = [int(landmarks.landmark[i].y * h) for i in CHEEK_POINTS]

    x_min, x_max = max(0, min(xs)), min(w, max(xs))
    y_min, y_max = max(0, min(ys)), min(h, max(ys))

    roi = frame[y_min:y_max, x_min:x_max]
    return roi if roi.size > 0 else None
