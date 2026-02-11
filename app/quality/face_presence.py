from app.core.landmarks import get_face_landmarks

def evaluate_face_presence(frames):
    detected = 0

    for frame in frames:
        if get_face_landmarks(frame):
            detected += 1

    ratio = detected / len(frames) if frames else 0

    if ratio < 0.6:
        return {
            "score": 30,
            "status": "Poor",
            "message": "Face frequently not detected"
        }

    if ratio < 0.85:
        return {
            "score": 70,
            "status": "Partial",
            "message": "Face intermittently detected"
        }

    return {
        "score": 95,
        "status": "Good",
        "message": "Face clearly detected"
    }
