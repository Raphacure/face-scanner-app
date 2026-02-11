import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import mediapipe as mp
import numpy as np

_face_mesh = None

def get_face_mesh():
    global _face_mesh

    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    return _face_mesh


# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def get_face_landmarks(frame):
    face_mesh = get_face_mesh()

    results = face_mesh.process(frame)

    if not results.multi_face_landmarks:
        return None

    return results.multi_face_landmarks[0]


def eye_aspect_ratio(landmarks, eye_indices):
    points = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in eye_indices]

    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (v1 + v2) / (2.0 * h)
