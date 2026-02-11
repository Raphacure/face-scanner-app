import numpy as np

def sdnn(rr_intervals):
    """Standard deviation of NN intervals (ms)"""
    if len(rr_intervals) < 2:
        return 0
    return float(np.std(rr_intervals) * 1000)


def rmssd(rr_intervals):
    """Root mean square of successive differences (ms)"""
    if len(rr_intervals) < 2:
        return 0
    diff = np.diff(rr_intervals)
    return float(np.sqrt(np.mean(diff ** 2)) * 1000)


def pnn50(rr_intervals):
    """Percentage of successive RR intervals differing > 50 ms"""
    if len(rr_intervals) < 2:
        return 0
    diff = np.abs(np.diff(rr_intervals) * 1000)
    return float(np.sum(diff > 50) / len(diff) * 100)
