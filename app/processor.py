from datetime import datetime
import cv2
import mediapipe as mp
from app.core.frame_buffer import add_frame, is_ready, get_frames, clear, count
from app.core.aggregator import calculate_all
from app.quality.quality_aggregator import evaluate_scan_quality

from app.pdf.report_generator import generate_health_report
from app.aws.s3_uploader import upload_pdf_to_s3
import os
from app.whatsapp.send_whatsapp import send_whatsapp_pdf


mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection()


def process_video_frames(frame, scan_id):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detector.process(rgb)

    if not results.detections:
        return {
            "status": "error",
            "message": "No face detected"
        }

    # ✅ Add frame to buffer
    add_frame(scan_id, rgb)

    # ✅ Still collecting frames
    if not is_ready(scan_id):
        return {
            "status": "processing",
            "message": f"Collecting frames {count(scan_id)}/300"
        }

    # ✅ Scan Complete: Process all frames
    frames = get_frames(scan_id)

    quality = evaluate_scan_quality(frames)
    data = calculate_all(frames)


    # ✅ Reduce confidence based on quality
    if quality["level"] == "Low":
        for group in data.values():
            for metric in group.values():
                metric["confidence"] *= 0.6

    elif quality["level"] == "Medium":
        for group in data.values():
            for metric in group.values():
                metric["confidence"] *= 0.8

    # ✅ Clear memory buffer
    clear(scan_id)

    # -------------------------------------------------
    # ✅ AUTO GENERATE PDF REPORT + UPLOAD TO S3
    # -------------------------------------------------

    # Temporary local file
    filename = f"/tmp/report_{scan_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"

    user = {
        "name": "Dileep",
        "age": 25,
        "gender": "Male"
    }

    # ✅ Generate PDF
    generate_health_report(user, data, filename)

    # ✅ Upload to AWS S3
    report_url = upload_pdf_to_s3(filename)

    # ✅ Delete local file after upload
    if os.path.exists(filename):
        os.remove(filename)

    # -------------------------------------------------
    # ✅ FINAL RESPONSE
    # -------------------------------------------------

    # sending whats report
    send_whatsapp_pdf("917337529401",report_url)
    return {
        "status": "success",
        "quality": quality,
        "data": data,
        "reportUrl": report_url   # ✅ Important
    }