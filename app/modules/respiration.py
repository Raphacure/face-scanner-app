from app.core.units import UNITS


class RespirationMetric:
    name = "respiration"

    def calculate(self, context):
        hr = context["results"].get("heartRate", {}).get("value", 0)

        resp = int(hr / 4)

        return {
            "value": resp,
            "unit": UNITS[self.name],
            "confidence": 0.6,
            "interpretation": "High Respiration" if resp > 15 else "Normal"
        }
