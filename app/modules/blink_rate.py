from app.core import units
from app.core.landmarks import get_face_landmarks, eye_aspect_ratio, LEFT_EYE, RIGHT_EYE
from app.core.units import UNITS

class BlinkRateMetric:
    name = "blinkRate"

    def calculate(self, context):
        frames = context["frames"]
        fps = context.get("fps", 30)

        blink_count = 0
        closed = False
        EAR_THRESHOLD = 0.21

        for frame in frames:
            landmarks = get_face_landmarks(frame)
            if not landmarks:
                continue

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            if ear < EAR_THRESHOLD and not closed:
                blink_count += 1
                closed = True
            elif ear >= EAR_THRESHOLD:
                closed = False

        duration_minutes = len(frames) / fps / 60
        blink_rate = int(blink_count / duration_minutes) if duration_minutes > 0 else 0

        return {
            "value": blink_rate,
            "unit": UNITS[self.name],
            "confidence": 0.75,
            "interpretation": (
                "Low" if blink_rate < 10 else
                "Normal" if blink_rate <= 20 else
                "High"
            )
        }
