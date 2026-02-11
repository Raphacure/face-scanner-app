from app.core.units import UNITS
import numpy as np
from app.core.landmarks import get_face_landmarks

class MotionStabilityMetric:
    name = "motionStability"

    def calculate(self, context):
        frames = context["frames"]

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
                dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                movements.append(dist)

            prev_center = center

        avg_motion = np.mean(movements) if movements else 0

        stability = max(0, min(100, int(100 - avg_motion * 800)))

        return {
            "value": stability,
            "unit": UNITS[self.name],
            "confidence": 0.8,
            "interpretation": (
                "Stable" if stability > 70 else "Unstable"
            )
        }
