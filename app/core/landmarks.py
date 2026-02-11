import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices (MediaPipe standard)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def get_face_landmarks(frame):
    results = face_mesh.process(frame)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]


def eye_aspect_ratio(landmarks, eye_indices):
    points = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in eye_indices]

    # vertical distances
    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))

    # horizontal distance
    h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (v1 + v2) / (2.0 * h)
