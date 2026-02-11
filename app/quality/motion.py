import numpy as np
from app.core.landmarks import get_face_landmarks

def evaluate_motion(frames):
    movements = []
    prev_center = None

    for frame in frames:
        landmarks = get_face_landmarks(frame)
        if not landmarks:
            continue

        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]

        center = (np.mean(xs), np.mean(ys))

        if prev_center:
            movements.append(np.linalg.norm(
                np.array(center) - np.array(prev_center)
            ))

        prev_center = center

    avg_motion = np.mean(movements) if movements else 0

    if avg_motion > 0.03:
        return {
            "score": 40,
            "status": "Unstable",
            "message": "Excessive head movement"
        }

    if avg_motion > 0.015:
        return {
            "score": 70,
            "status": "Moderate",
            "message": "Some movement detected"
        }

    return {
        "score": 90,
        "status": "Stable",
        "message": "Good stability"
    }
