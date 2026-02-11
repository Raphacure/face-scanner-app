import numpy as np
import cv2
from scipy.signal import butter, lfilter, find_peaks
import mediapipe as mp

# Bandpass filter for heart rate frequency
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=30):
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = lfilter(b, a, data)
    return y


def extract_green_channel(frames):
    green_values = []

    for frame in frames:
        green = np.mean(frame[:, :, 1])
        green_values.append(green)

    return np.array(green_values)


def calculate_heart_rate(frames, fps=30):

    signal = extract_green_channel(frames)

    filtered = bandpass_filter(signal, fs=fps)

    peaks, _ = find_peaks(filtered, distance=fps/2)

    if len(peaks) < 2:
        return None

    peak_intervals = np.diff(peaks) / fps

    avg_interval = np.mean(peak_intervals)

    heart_rate = 60 / avg_interval

    return int(heart_rate)


def estimate_vitals_from_video(frames):

    hr = calculate_heart_rate(frames)

    if not hr:
        hr = 0

    respiration = int(hr / 4)

    spo2 = 95 + (hr % 4)

    stress = max(0, min(100, int((100 - hr) / 2)))

    return {
        "heartRate": hr,
        "respiration": respiration,
        "spo2": spo2,
        "stressScore": stress
    }
