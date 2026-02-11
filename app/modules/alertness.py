from app.core.units import UNITS


class AlertnessMetric:
    name = "alertness"

    def calculate(self, context):
        fatigue = context["results"].get("fatigue", {}).get("value", 50)

        alertness = max(0, min(100, 100 - fatigue))

        return {
    "value": alertness,
    "unit": UNITS[self.name],
    "confidence": 0.7,
    "interpretation": "High Alertness" if alertness > 70 else "Normal"
}
