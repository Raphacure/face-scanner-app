from app.skin.redness import SkinRednessMetric
from app.skin.texture import SkinTextureMetric
from app.skin.hydration import SkinHydrationMetric
from app.skin.dark_circles import DarkCirclesMetric

SKIN_METRICS = [
    SkinRednessMetric(),
    SkinTextureMetric(),
    SkinHydrationMetric(),
    DarkCirclesMetric()
]

def calculate_skin_health(context):
    results = {}

    for metric in SKIN_METRICS:
        results[metric.name] = metric.calculate(context)

    values = [
        results[m]["value"]
        for m in results
        if results[m]["confidence"] > 0
    ]

    skin_score = int(sum(values) / len(values)) if values else 0

    print("skin_score",skin_score)

    results["skinHealthScore"] = {
        "value": skin_score,
        "unit": "%",
        "confidence": 0.7,
        "interpretation": (
            "Good" if skin_score > 70 else
            "Moderate" if skin_score > 40 else
            "Poor"
        )
    }

    return results
