import cv2
import numpy as np
from app.skin.face_roi import extract_face_roi

class SkinTextureMetric:
    name = "skinTexture"

    def calculate(self, context):
        frames = context["frames"]
        texture_vals = []

        for frame in frames:
            roi = extract_face_roi(frame)
            if roi is None:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            texture_vals.append(np.var(lap))

        if not texture_vals:
            return {
                "value": 0,
                "unit": "%",
                "confidence": 0.0,
                "interpretation": "Insufficient data"
            }

        avg_var = np.mean(texture_vals)
        smoothness = int(max(0, min(100, 100 - avg_var / 5)))

        interpretation = (
            "Smooth" if smoothness > 70 else
            "Normal" if smoothness > 40 else
            "Rough"
        )

        return {
            "value": smoothness,
            "unit": "%",
            "confidence": 0.75,
            "interpretation": interpretation
        }
