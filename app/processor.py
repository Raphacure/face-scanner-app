from ast import If
import os
import gc
from datetime import datetime

# ✅ MUST be before mediapipe import
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from app.services.send_email import send_email
from app.services.send_whatsapp import send_whatsapp_pdf
import cv2
import mediapipe as mp

from app.core.frame_buffer import add_frame, is_ready, get_frames, clear, count
from app.core.aggregator import calculate_all
from app.quality.quality_aggregator import evaluate_scan_quality
from app.pdf.report_generator import generate_health_report, generate_health_report_v2, get_user_details, insert_face_scan
from app.aws.s3_uploader import upload_image_to_s3, upload_pdf_to_s3


# ✅ Create ONCE per worker
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

def process_video_frames(request, frame, scan_id, userId, clientId):

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
            "message": f"Collecting frames {count(scan_id)}/100",
            "count": count(scan_id)
        }

    frames = get_frames(scan_id)

    quality = evaluate_scan_quality(frames)


    clear(scan_id)

    image_path = f"/tmp/scan_{scan_id}.jpg"
    filename = f"/tmp/report_{scan_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    filename_v2 = f"/tmp/report_v2_{scan_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"

    try:

        # --------------------------------------------------
        # SAVE IMAGE
        # --------------------------------------------------
        cv2.imwrite(image_path, frame)

        # Upload image to S3
        image_url = upload_image_to_s3(image_path, scan_id)

        # --------------------------------------------------
        # GET USER DETAILS
        # --------------------------------------------------
        try:
            user_details = get_user_details(userId)
        except Exception as e:
            print("error", e)
            user_details = None

        users = user_details.get("user", []) if user_details else []
        userobj = users[0] if users else {}

        first_name = userobj.get("first_name")
        last_name = userobj.get("last_name")

        user = {
            "name": f"{first_name} {last_name}" if first_name and last_name else "N/A",
            "age": userobj.get("age") if userobj.get("age") is not None else '',
            "gender": userobj.get("gender") if userobj.get("gender") else "N/A",
        }

        email = userobj.get("email")
        raw_phone = userobj.get("phone")

        phone = None
        if raw_phone:
            phone = f"91{raw_phone}"

        print("phone number", phone)

        data = calculate_all(frames,user)
        # --------------------------------------------------
        # GENERATE PDFS (KEEP EXISTING + ADD V2)
        # --------------------------------------------------
        generate_health_report(user, data, filename,image_path)
        generate_health_report_v2(user, data, filename_v2, face_image_path=image_path)

        # Upload PDF
        report_url = upload_pdf_to_s3(filename)
        report_v2_url = upload_pdf_to_s3(filename_v2)

        # --------------------------------------------------
        # DEVICE + IP
        # --------------------------------------------------
        ip_address = request.headers.get("x-forwarded-for")
        if ip_address:
            ip_address = ip_address.split(",")[0].strip()
        else:
            ip_address = request.client.host

        user_agent = request.headers.get("user-agent", "").lower()

        if "mobile" in user_agent:
            device = "mobile"
        elif "android" in user_agent:
            device = "android"
        elif "iphone" in user_agent:
            device = "iphone"
        elif "windows" in user_agent or "macintosh" in user_agent:
            device = "desktop"
        else:
            device = "unknown"

        print("device", device)

        # --------------------------------------------------
        # SAVE SCAN DATA
        # --------------------------------------------------
        faceScanData = {
            "device": device,
            "ip": ip_address,
            "user_id": userId,
            "client_id": clientId,
            "report_url": [report_url, report_v2_url],
            "image_url": image_url,
            "response": data
        }

        insert_face_scan(faceScanData)

        # Send WhatsApp
        if phone:
            send_whatsapp_pdf(phone, report_url, report_v2_url)
        
        if email:
            html_content = f"""
            <p>Dear {first_name or ''} {last_name or ''},</p>
            <p>We are pleased to inform you that your face scan has been successfully completed.</p>
            <p>Please find your detailed report attached to this email in PDF format.</p>
            <p>If you have any questions, feel free to reach out.</p>
            <p>Thank you for using our service.</p>
            """ 
            send_email(email, "Health Assessment Report", html_content, [report_url, report_v2_url])

        return {
            "status": "success",
            "quality": quality,
            "data": data,
            "report_url": [report_url, report_v2_url],
            "image_url": image_url
        }

    except Exception as e:

        print("Face scan processing error:", e)

        return {
            "status": "error",
            "message": "Scan processing failed"
        }

    finally:

        # --------------------------------------------------
        # ALWAYS DELETE TEMP FILES
        # --------------------------------------------------
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except:
            pass

        try:
            if os.path.exists(filename):
                os.remove(filename)
        except:
            pass
        try:
            if os.path.exists(filename_v2):
                os.remove(filename_v2)
        except:
            pass

        # Memory cleanup
        try:
            del frames
            del rgb
            del small
        except:
            pass

        gc.collect()