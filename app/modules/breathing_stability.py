from app.core.units import UNITS


class BreathingStabilityMetric:
    name = "breathingStability"

    def calculate(self, context):
        respiration = context["results"].get("respiration", {}).get("value", 0)

        if respiration == 0:
            return {"value": 0, "confidence": 0.0}

        # Optimal breathing range: 12–18 bpm
        deviation = abs(respiration - 15)

        stability = max(0, min(100, int(100 - deviation * 7)))

        return {
    "value": stability,
    "unit": UNITS[self.name],
    "confidence": 0.6,
    "interpretation": "Stable" if stability > 70 else "Unstable"
}
