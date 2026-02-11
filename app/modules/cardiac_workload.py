from app.core.units import UNITS


class CardiacWorkloadMetric:
    name = "cardiacWorkload"

    def calculate(self, context):
        hr = context["results"].get("heartRate", {}).get("value", 0)
        hrv = context["results"].get("hrv", {}).get("value", 0)

        if hr == 0 or hrv == 0:
            return {"value": 0, "confidence": 0.0}

        workload = hr * (1 / (hrv + 1))

        workload_score = min(100, int(workload * 10))

        return {
            "value": workload_score,
            "unit": UNITS[self.name],
            "confidence": 0.75,
            "interpretation": "High Cardiac Workload" if workload_score > 70 else "Normal"
        }
