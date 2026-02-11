class WellnessMetric:
    name = "wellnessScore"

    def calculate(self, context):

        hr = context["results"].get("heartRate", {}).get("value", 70)
        hrv = context["results"].get("hrv", {}).get("value", 40)
        stress = context["results"].get("stress", {}).get("value", 50)

        # ---------------------------
        # Score Formula
        # ---------------------------
        score = int(
            (hrv * 0.4) +
            ((100 - stress) * 0.3) +
            ((100 - abs(70 - hr)) * 0.3)
        )

        score = max(0, min(100, score))

        # ---------------------------
        # Interpretation / Status
        # ---------------------------
        if score >= 80:
            status = "Excellent"
        elif score >= 65:
            status = "Good"
        elif score >= 50:
            status = "Moderate"
        else:
            status = "Poor"

        return {
            "value": score,
            "unit": "/100",
            "confidence": 0.8,
            "interpretation": status
        }
