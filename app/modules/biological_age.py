from app.core.units import UNITS


class BiologicalAge:

    name = "biologicalAge"

    def calculate(self, results):

        data = results.get("results", {})
        print("result-data:", results.get("user", {}))

        # -------------------------
        # CARDIO METRICS
        # -------------------------

        hrv = data.get("hrv", {}).get("value", 40)
        spo2 = data.get("spo2", {}).get("value", 98)
        respiration = data.get("respiration", {}).get("value", 16)
        pulse_regularity = data.get("pulseRegularity", {}).get("value", 90)


        # -------------------------
        # STRESS METRICS
        # -------------------------

        stress = data.get("stress", {}).get("value", 50)
        fatigue = data.get("fatigue", {}).get("value", 50)
        recovery = data.get("recoveryIndex", {}).get("value", 50)



        # -------------------------
        # SKIN METRICS
        # -------------------------

        dark_circles = data.get("skin",{}).get("darkCircles", {}).get("value", 50)
        hydration = data.get("skin",{}).get("skinHydration", {}).get("value", 50)
        texture = data.get("skin",{}).get("skinTexture", {}).get("value", 50)

        # -------------------------
        # CARDIO SCORE
        # -------------------------

        cardio_score = (
            (hrv * 0.25) +
            (pulse_regularity * 0.2) +
            (spo2 * 0.2) -
            (respiration * 0.1)
        )


        # -------------------------
        # STRESS SCORE
        # -------------------------

        stress_score = (
            (stress * 0.3) +
            (fatigue * 0.3) -
            (recovery * 0.2)
        )


        # -------------------------
        # SKIN SCORE
        # -------------------------

        skin_score = (
            (dark_circles * 0.25) -
            (hydration * 0.2) -
            (texture * 0.2)
        )


        # -------------------------
        # FINAL HEALTH SCORE
        # -------------------------

        health_score = cardio_score + stress_score + skin_score

        user = results.get("user", {})

        age = user.get("age")

        if isinstance(age, (int, float)):
            base_age = age
        else:
            base_age = 28

        print("base_age:", base_age)

        age_shift = health_score * 0.05
        print("Age Shift:", age_shift)

        biological_age = base_age + age_shift
        biological_age = max(15, min(90, biological_age))

        print("Final Biological Age:", biological_age)

        return {
            "value": round(biological_age),
            "unit": UNITS[self.name],
            "confidence": 0.82,
        }