import numpy as np
import cv2
from app.skin.face_roi import extract_face_roi

class SkinHydrationMetric:
    name = "skinHydration"

    def calculate(self, context):
        frames = context["frames"]
        hydration_vals = []

        for frame in frames:
            roi = extract_face_roi(frame)
            if roi is None:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            hydration_vals.append(np.std(gray))

        if not hydration_vals:
            return {
                "value": 0,
                "unit": "%",
                "confidence": 0.0,
                "interpretation": "Insufficient data"
            }

        avg = np.mean(hydration_vals)
        hydration = int(max(0, min(100, 100 - avg)))

        interpretation = (
            "Well Hydrated" if hydration > 70 else
            "Moderate" if hydration > 40 else
            "Dry"
        )

        return {
            "value": hydration,
            "unit": "%",
            "confidence": 0.6,
            "interpretation": interpretation
        }
