from app.quality.lighting import evaluate_lighting
from app.quality.motion import evaluate_motion
from app.quality.face_presence import evaluate_face_presence

def evaluate_scan_quality(frames):
    lighting = evaluate_lighting(frames)
    motion = evaluate_motion(frames)
    face = evaluate_face_presence(frames)

    overall_score = int(
        lighting["score"] * 0.4 +
        motion["score"] * 0.4 +
        face["score"] * 0.2
    )

    if overall_score < 50:
        level = "Low"
    elif overall_score < 75:
        level = "Medium"
    else:
        level = "High"

    return {
        "overallScore": overall_score,
        "level": level,
        "components": {
            "lighting": lighting,
            "motion": motion,
            "facePresence": face
        }
    }
