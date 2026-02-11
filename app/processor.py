import os
import gc
from datetime import datetime

# ✅ MUST be before mediapipe import
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import mediapipe as mp

from app.core.frame_buffer import add_frame, is_ready, get_frames, clear, count
from app.core.aggregator import calculate_all
from app.quality.quality_aggregator import evaluate_scan_quality
from app.pdf.report_generator import generate_health_report
from app.aws.s3_uploader import upload_pdf_to_s3
# from app.whatsapp.send_whatsapp import send_whatsapp_pdf


# ✅ Create ONCE per worker
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


def process_video_frames(frame, scan_id):

    # ✅ Resize FIRST → huge CPU & RAM savings
    small = cv2.resize(frame, (320, 240))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    results = face_detector.process(rgb)

    if not results.detections:
        return {
            "status": "error",
            "message": "No face detected"
        }

    add_frame(scan_id, rgb)

    if not is_ready(scan_id):
        return {
            "status": "processing",
            "message": f"Collecting frames {count(scan_id)}/300"
        }

    frames = get_frames(scan_id)

    quality = evaluate_scan_quality(frames)
    data = calculate_all(frames)

    clear(scan_id)

    filename = f"/tmp/report_{scan_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"

    user = {
        "name": "Dileep",
        "age": 25,
        "gender": "Male"
    }

    generate_health_report(user, data, filename)
    report_url = upload_pdf_to_s3(filename)

    if os.path.exists(filename):
        os.remove(filename)

    # ✅ Strong cleanup for long-running workers
    del frames
    del rgb
    del small
    gc.collect()

    # send_whatsapp_pdf("917337529401", report_url)

    return {
        "status": "success",
        "quality": quality,
        "data": data,
        "report_url": report_url
    }
