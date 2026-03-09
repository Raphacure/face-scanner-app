def calculate_dosha(context):

    results = context.get("results", {})

    hrv = results.get("hrv", {}).get("value", 0)
    heart_rate = results.get("heartRate", {}).get("value", 0)

    stress = results.get("stress", {}).get("value", 0)
    workload = results.get("cardiacWorkload", {}).get("value", 0)

    relaxation = results.get("relaxation", {}).get("value", 0)
    recovery = results.get("recoveryIndex", {}).get("value", 0)

    motion = results.get("motionStability", {}).get("value", 50)
    pulse = results.get("pulseRegularity", {}).get("value", 50)
    breathing = results.get("breathingStability", {}).get("value", 50)

    # Dosha scores
    vata = (hrv * 0.7) + ((100 - heart_rate) * 0.3)
    pitta = (stress * 0.6) + (workload * 0.4)
    kapha = (relaxation * 0.6) + (recovery * 0.4)

    total = vata + pitta + kapha or 1

    vata_pct = round(vata / total * 100)
    pitta_pct = round(pitta / total * 100)
    kapha_pct = round(kapha / total * 100)

    dominant = max(
        {"vata": vata_pct, "pitta": pitta_pct, "kapha": kapha_pct},
        key=lambda k: {"vata": vata_pct, "pitta": pitta_pct, "kapha": kapha_pct}[k]
    )

    # Confidence based on signal stability
    confidence = round((motion + pulse + breathing) / 300, 2)

    return {
        "vata": {
            "value": vata_pct,
            "unit": "%",
            "confidence": confidence,
            "interpretation": "Dominant" if dominant == "vata" else "Secondary"
        },
        "pitta": {
            "value": pitta_pct,
            "unit": "%",
            "confidence": confidence,
            "interpretation": "Dominant" if dominant == "pitta" else "Secondary"
        },
        "kapha": {
            "value": kapha_pct,
            "unit": "%",
            "confidence": confidence,
            "interpretation": "Dominant" if dominant == "kapha" else "Secondary"
        }
    }