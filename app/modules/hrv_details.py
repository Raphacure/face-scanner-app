import numpy as np
from app.core.signal_processor import extract_green_signal, bandpass_filter, detect_peaks
from app.core.hrv_utils import sdnn, rmssd, pnn50

class HRVDetailsMetric:
    name = "hrvDetails"

    def calculate(self, context):
        frames = context["frames"]
        fps = context.get("fps", 30)

        signal = extract_green_signal(frames)
        filtered = bandpass_filter(signal, fs=fps)
        peaks = detect_peaks(filtered, fps)

        if len(peaks) < 3:
            return {
                "value": {
                    "sdnn": 0,
                    "rmssd": 0,
                    "pnn50": 0
                },
                "unit": {
                    "sdnn": "ms",
                    "rmssd": "ms",
                    "pnn50": "%"
                },
                "confidence": 0.0,
                "interpretation": "Insufficient data"
            }

        rr_intervals = np.diff(peaks) / fps

        sdnn_val = round(sdnn(rr_intervals), 1)
        rmssd_val = round(rmssd(rr_intervals), 1)
        pnn50_val = round(pnn50(rr_intervals), 1)

        interpretation = (
            "Good" if rmssd_val > 30 else
            "Moderate" if rmssd_val > 15 else
            "Low"
        )

        return {
            "value": {
                "sdnn": sdnn_val,
                "rmssd": rmssd_val,
                "pnn50": pnn50_val
            },
            "unit": {
                "sdnn": "ms",
                "rmssd": "ms",
                "pnn50": "%"
            },
            "confidence": 0.8,
            "interpretation": interpretation
        }
