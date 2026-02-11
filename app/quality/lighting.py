import numpy as np
import cv2

def evaluate_lighting(frames):
    brightness_values = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness_values.append(np.mean(gray))

    avg_brightness = np.mean(brightness_values)

    if avg_brightness < 60:
        return {
            "score": 30,
            "status": "Poor",
            "message": "Low lighting detected"
        }

    if avg_brightness < 100:
        return {
            "score": 60,
            "status": "Fair",
            "message": "Lighting could be improved"
        }

    return {
        "score": 90,
        "status": "Good",
        "message": "Lighting is good"
    }
