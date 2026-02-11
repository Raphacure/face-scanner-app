from app.core.units import UNITS
import numpy as np
from app.core.signal_processor import extract_green_signal, bandpass_filter, detect_peaks

class HRVMetric:
    name = "hrv"

    def calculate(self, context):
        frames = context["frames"]
        fps = context.get("fps", 30)

        signal = extract_green_signal(frames)
        filtered = bandpass_filter(signal, fs=fps)

        peaks = detect_peaks(filtered, fps)

        if len(peaks) < 3:
            return {"value": 0, "confidence": 0.0}

        intervals = np.diff(peaks) / fps

        hrv = float(np.std(intervals) * 100)

        return {
            "value": round(hrv, 2),
            "unit": UNITS[self.name],
            "confidence": 0.8,
            "interpretation": "High HRV" if hrv > 40 else "Normal"
        }
