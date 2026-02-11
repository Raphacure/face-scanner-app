from app.core.units import UNITS


class RecoveryIndexMetric:
    name = "recoveryIndex"

    def calculate(self, context):
        hrv = context["results"].get("hrv", {}).get("value", 0)
        hr = context["results"].get("heartRate", {}).get("value", 0)

        if hr == 0:
            return {"value": 0, "confidence": 0.0}

        recovery = max(0, min(100, int((hrv * 1.2) - (hr * 0.3))))

        return {
            "value": recovery,
            "unit": UNITS[self.name],
            "confidence": 0.75,
            "interpretation": "High Recovery" if recovery > 70 else "Normal"
        }
