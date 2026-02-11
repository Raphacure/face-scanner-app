from app.core.units import UNITS
from app.core.interpretation import interpret_range


class StressMetric:
    name = "stress"

    def calculate(self, context):
        hr = context["results"].get("heartRate", {}).get("value", 0)
        hrv = context["results"].get("hrv", {}).get("value", 0)
        respiration = context["results"].get("respiration", {}).get("value", 0)

        if hr == 0 or hrv == 0:
            return {
                "value": 0,
                "unit": "%",
                "confidence": 0.0,
                "interpretation": "Insufficient data"
            }

        # Stress model (heuristic but realistic)
        stress = (
            (hr - 60) * 0.6 +
            (40 - hrv) * 0.8 +
            (respiration - 12) * 0.4
        )

        stress = int(max(0, min(100, stress)))

        interpretation = interpret_range(
            stress,
            {
                "Low": (0, 30),
                "Moderate": (31, 60),
                "High": (61, 100)
            }
        )

        return {
            "value": stress,
            "unit": "%",
            "confidence": 0.75,
            "interpretation": interpretation
        }
