from app.core.units import UNITS


class FatigueMetric:
    name = "fatigue"

    def calculate(self, context):
        hr = context["results"].get("heartRate", {}).get("value", 70)
        hrv = context["results"].get("hrv", {}).get("value", 40)

        fatigue = max(0, min(100, 100 - (hrv + hr/2)))

        return {
    "value": fatigue,
    "unit": UNITS[self.name],
    "confidence": 0.7,
    "interpretation": "High Fatigue" if fatigue > 60 else "Normal"
}
