from app.core.units import UNITS
import numpy as np
from app.core.signal_processor import extract_green_signal, bandpass_filter, detect_peaks

class PulseRegularityMetric:
    name = "pulseRegularity"

    def calculate(self, context):
        frames = context["frames"]
        fps = context.get("fps", 30)

        signal = extract_green_signal(frames)
        filtered = bandpass_filter(signal, fs=fps)
        peaks = detect_peaks(filtered, fps)

        if len(peaks) < 3:
            return {"value": 0, "confidence": 0.0}

        intervals = np.diff(peaks) / fps
        std_dev = np.std(intervals)

        # Convert to score (lower std = better regularity)
        regularity = max(0, min(100, int(100 - (std_dev * 120))))

        return {
            "value": regularity,
            "unit": UNITS[self.name],
            "confidence": 0.8,
            "interpretation": "High Pulse Regularity" if regularity > 70 else "Normal"
        }
