from app.core.units import UNITS


class RelaxationMetric:
    name = "relaxationScore"

    def calculate(self, context):
        fatigue = context["results"].get("fatigue", {}).get("value", 50)
        alertness = context["results"].get("alertness", {}).get("value", 50)

        relaxation = max(0, min(100, int((alertness + (100 - fatigue)) / 2)))

        return {
            "value": relaxation,
            "unit": UNITS[self.name],
            "confidence": 0.7,
            "interpretation": "High Relaxation" if relaxation > 70 else "Normal"
        }
