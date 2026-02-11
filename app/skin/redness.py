import numpy as np
import cv2
from app.skin.face_roi import extract_face_roi

class SkinRednessMetric:
    name = "skinRedness"

    def calculate(self, context):
        frames = context["frames"]
        redness_values = []

        for frame in frames:
            roi = extract_face_roi(frame)
            if roi is None:
                continue

            r, g, b = cv2.split(roi.astype("float"))
            redness = np.mean(r / (g + 1))
            redness_values.append(redness)

        if not redness_values:
            return {
                "value": 0,
                "unit": "%",
                "confidence": 0.0,
                "interpretation": "Insufficient data"
            }

        avg = np.mean(redness_values)
        redness_score = int(min(100, max(0, (avg - 1) * 50)))

        interpretation = (
            "Low" if redness_score < 30 else
            "Moderate" if redness_score < 60 else
            "High"
        )

        return {
            "value": redness_score,
            "unit": "%",
            "confidence": 0.7,
            "interpretation": interpretation
        }
