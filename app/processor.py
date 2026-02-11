import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"   # ✅ FIRST LINE

import cv2
import mediapipe as mp

from datetime import datetime

import gc
import os

from app.core.frame_buffer import add_frame, is_ready, get_frames, clear, count
from app.core.aggregator import calculate_all
from app.quality.quality_aggregator import evaluate_scan_quality
from app.pdf.report_generator import generate_health_report
from app.aws.s3_uploader import upload_pdf_to_s3
from app.whatsapp.send_whatsapp import send_whatsapp_pdf


def get_face_detector():
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )


def process_video_frames(frame, scan_id):

    detector = get_face_detector()

    small = cv2.resize(frame, (320, 240))   # ✅ CRITICAL FIX
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    results = detector.process(rgb)

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

    if quality["level"] == "Low":
        for group in data.values():
            for metric in group.values():
                metric["confidence"] *= 0.6

    elif quality["level"] == "Medium":
        for group in data.values():
            for metric in group.values():
                metric["confidence"] *= 0.8

    clear(scan_id)

    # filename = f"/tmp/report_{scan_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"

    # user = {
    #     "name": "Dileep",
    #     "age": 25,
    #     "gender": "Male"
    # }

    # generate_health_report(user, data, filename)
    # report_url = upload_pdf_to_s3(filename)

    # if os.path.exists(filename):
    #     os.remove(filename)

    # send_whatsapp_pdf("917337529401", report_url)

    # ✅ Force memory cleanup (IMPORTANT FOR EC2)
    del frames
    del rgb
    gc.collect()

    return {
        "status": "success",
        "quality": quality,
        "data": data,
        # "reportUrl": report_url
    }
