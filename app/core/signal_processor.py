import numpy as np
from scipy.signal import butter, lfilter, find_peaks


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=30):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)


def extract_green_signal(frames):
    values = [np.mean(f[:, :, 1]) for f in frames]
    return np.array(values)


def detect_peaks(signal, fps=30):
    peaks, _ = find_peaks(signal, distance=fps/2)
    return peaks
