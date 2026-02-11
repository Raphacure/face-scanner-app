from app.core.landmarks import get_face_landmarks, eye_aspect_ratio, LEFT_EYE, RIGHT_EYE
from app.core.units import UNITS

class EyeClosureMetric:
    name = "eyeClosureDuration"

    def calculate(self, context):
        frames = context["frames"]
        fps = context.get("fps", 30)

        closed_frames = 0
        EAR_THRESHOLD = 0.21

        for frame in frames:
            landmarks = get_face_landmarks(frame)
            if not landmarks:
                continue

            ear = (
                eye_aspect_ratio(landmarks, LEFT_EYE) +
                eye_aspect_ratio(landmarks, RIGHT_EYE)
            ) / 2

            if ear < EAR_THRESHOLD:
                closed_frames += 1

        closure_seconds = closed_frames / fps

        return {
            "value": round(closure_seconds, 2),
            "unit": UNITS[self.name],
            "confidence": 0.7,
            "interpretation": (
                "Normal" if closure_seconds < 1.5 else
                "Drowsy"
            )
        }
