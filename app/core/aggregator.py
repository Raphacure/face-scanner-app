from app.modules.heart_rate import HeartRateMetric
from app.modules.hrv import HRVMetric
from app.modules.hrv_details import HRVDetailsMetric
from app.modules.respiration import RespirationMetric
from app.modules.spo2 import SpO2Metric
from app.modules.stress import StressMetric

from app.modules.blink_rate import BlinkRateMetric
from app.modules.eye_closure import EyeClosureMetric
from app.modules.motion_stability import MotionStabilityMetric

from app.modules.fatigue import FatigueMetric
from app.modules.alertness import AlertnessMetric

from app.modules.pulse_regularity import PulseRegularityMetric
from app.modules.cardiac_workload import CardiacWorkloadMetric
from app.modules.breathing_stability import BreathingStabilityMetric
from app.modules.relaxation import RelaxationMetric
from app.modules.recovery import RecoveryIndexMetric

from app.modules.wellness import WellnessMetric

from app.skin.skin_aggregator import calculate_skin_health
from app.core.group_results import group_metrics


# ✅ METRICS MUST LIVE IN THIS FILE
METRICS = [
    HeartRateMetric(),
    HRVMetric(),
    HRVDetailsMetric(),
    RespirationMetric(),
    SpO2Metric(),
    StressMetric(),

    BlinkRateMetric(),
    EyeClosureMetric(),
    MotionStabilityMetric(),

    FatigueMetric(),
    AlertnessMetric(),

    PulseRegularityMetric(),
    CardiacWorkloadMetric(),
    BreathingStabilityMetric(),
    RelaxationMetric(),
    RecoveryIndexMetric(),

    WellnessMetric()
]


def calculate_all(frames):
    context = {
        "frames": frames,
        "fps": 30,
        "results": {}
    }

    flat_results = {}

    # 1️⃣ Run all metric modules
    for metric in METRICS:
        value = metric.calculate(context)
        flat_results[metric.name] = value
        context["results"][metric.name] = value

    # 2️⃣ Run skin aggregation ONCE
    flat_results["skin"] = calculate_skin_health(context)

    # 3️⃣ Group final response
    return group_metrics(flat_results)
