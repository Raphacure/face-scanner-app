import numpy as np
from app.core.signal_processor import extract_green_signal, bandpass_filter
from app.core.units import UNITS
from app.core.interpretation import interpret_range


class SpO2Metric:
    name = "spo2"

    def calculate(self, context):
        frames = context["frames"]
        fps = context.get("fps", 30)

        if len(frames) < 60:
            return {
                "value": 0,
                "unit": "%",
                "confidence": 0.0,
                "interpretation": "Insufficient data"
            }

        # Extract green channel rPPG signal
        signal = extract_green_signal(frames)
        filtered = bandpass_filter(signal, fs=fps)

        # AC / DC components (simplified model)
        ac = np.std(filtered)
        dc = np.mean(signal)

        if dc == 0:
            spo2 = 0
            confidence = 0.0
        else:
            # Empirical estimation formula (wellness-grade)
            ratio = ac / dc
            spo2 = int(max(90, min(99, 100 - ratio * 20)))
            confidence = 0.65

        interpretation = interpret_range(
            spo2,
            {
                "Low": (0, 94),
                "Normal": (95, 100)
            }
        )

        return {
            "value": spo2,
            "unit": "%",
            "confidence": confidence,
            "interpretation": interpretation
        }
