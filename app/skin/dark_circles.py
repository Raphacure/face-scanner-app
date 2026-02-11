import numpy as np
import cv2
from app.skin.face_roi import extract_face_roi

class DarkCirclesMetric:
    name = "darkCircles"

    def calculate(self, context):
        frames = context["frames"]
        darkness_vals = []

        for frame in frames:
            roi = extract_face_roi(frame)
            if roi is None:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            darkness_vals.append(np.mean(gray))

        if not darkness_vals:
            return {
                "value": 0,
                "unit": "%",
                "confidence": 0.0,
                "interpretation": "Insufficient data"
            }

        avg = np.mean(darkness_vals)
        dark_score = int(max(0, min(100, 100 - avg)))

        interpretation = (
            "Low" if dark_score < 30 else
            "Moderate" if dark_score < 60 else
            "High"
        )

        return {
            "value": dark_score,
            "unit": "%",
            "confidence": 0.6,
            "interpretation": interpretation
        }
