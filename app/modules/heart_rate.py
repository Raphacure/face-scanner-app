from app.core.signal_processor import extract_green_signal, bandpass_filter, detect_peaks
from app.core.units import UNITS
from app.core.interpretation import interpret_range

class HeartRateMetric:
    name = "heartRate"

    def calculate(self, context):
        frames = context["frames"]
        fps = context.get("fps", 30)

        signal = extract_green_signal(frames)
        filtered = bandpass_filter(signal, fs=fps)
        peaks = detect_peaks(filtered, fps)

        if len(peaks) < 2:
            return {
                "value": 0,
                "unit": UNITS[self.name],
                "confidence": 0.0,
                "interpretation": "Insufficient data"
            }

        intervals = (peaks[1:] - peaks[:-1]) / fps
        hr = int(60 / intervals.mean())

        interpretation = interpret_range(
            hr,
            {
                "Low": (40, 59),
                "Normal": (60, 100),
                "Elevated": (101, 120),
                "High": (121, 200)
            }
        )

        return {
            "value": hr,
            "unit": UNITS[self.name],
            "confidence": 0.85,
            "interpretation": interpretation
        }
