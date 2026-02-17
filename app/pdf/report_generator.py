from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime
import requests

# ==========================================================
# ✅ NESTED METRIC GETTER
# ==========================================================
def get_metric(metrics, path, field="value", default="N/A"):
    """
    Supports nested keys:
    get_metric(metrics, "vitals.heartRate")
    get_metric(metrics, "heartHealth.cardiacWorkload", "interpretation")
    """
    try:
        keys = path.split(".")
        obj = metrics
        for k in keys:
            obj = obj[k]
        return obj.get(field, default)
    except:
        return default


# ==========================================================
# ✅ FIXED DESCRIPTIONS (Same PDF Style)
# ==========================================================
DESCRIPTIONS = {

    # ---------------- VITALS ----------------
    "vitals.heartRate":
        "The heart rate is the number of times the heart beats in a minute. "
        "A normal resting heart rate for adults ranges from 60 to 100 beats per minute.",

    "vitals.respiration":
        "The respiration rate is the number of breaths taken per minute. "
        "It is typically measured at rest by counting chest rises for one minute.",

    "vitals.spo2":
        "Oxygen saturation (SpO2) is the measurement of how much oxygen the blood "
        "is carrying as a percentage of the maximum it could carry.",


    # ---------------- HEART HEALTH ----------------
    "heartHealth.hrv":
        "Heart Rate Variability (HRV) reflects the variation in time between heartbeats. "
        "Higher HRV is generally associated with better recovery and stress resilience.",

    "heartHealth.pulseRegularity":
        "Pulse regularity indicates how consistent and steady your heartbeat rhythm is. "
        "Irregular patterns may suggest cardiovascular strain or fatigue.",

    "heartHealth.cardiacWorkload":
        "Risk due to heart workload strain indicates the potential for cardiovascular stress "
        "based on how hard the heart must work during daily activities or exercise.",

    "heartHealth.recoveryIndex":
        "The recovery index reflects how efficiently your cardiovascular system returns "
        "to baseline after exertion or stress.",


    # ---------------- MENTAL WELLNESS ----------------
    "mental.stress":
        "Body's response to everyday pressures resulting from emotional or physical tension. "
        "A stress index below 1.5 is considered normal.",

    "mental.fatigue":
        "Fatigue score reflects tiredness and reduced energy levels. "
        "Higher fatigue may indicate insufficient rest or prolonged stress.",

    "mental.relaxationScore":
        "Relaxation score indicates how calm and recovered your body is. "
        "Higher values suggest better relaxation and parasympathetic activity.",


    # ---------------- BEHAVIOR ----------------
    "behavior.blinkRate":
        "Blink rate is the number of times you blink per minute. "
        "An increased blink rate may indicate stress, fatigue, or eye strain.",

    "behavior.eyeClosureDuration":
        "Eye closure duration represents the average time your eyes remain closed. "
        "Longer closures may reflect tiredness or reduced alertness.",

    "behavior.motionStability":
        "Motion stability measures how steady you remained during the scan. "
        "Higher stability improves scan accuracy.",

    "behavior.alertness":
        "Alertness score reflects cognitive readiness and attentiveness. "
        "Higher alertness indicates better focus and awareness.",

        # ---------------- SKIN HEALTH ----------------
    "skin.skinRedness":
        "Skin redness reflects inflammation or irritation. "
        "Lower redness indicates healthier and calmer skin.",

    "skin.skinTexture":
        "Skin texture indicates smoothness and surface quality. "
        "Higher texture score suggests smoother and more even skin.",

    "skin.skinHydration":
        "Skin hydration measures moisture levels in the skin. "
        "Well hydrated skin appears healthier, plumper, and more elastic.",

    "skin.darkCircles":
        "Dark circles indicate pigmentation or shadowing under the eyes. "
        "Lower values suggest minimal under-eye darkness.",

    "skin.skinHealthScore":
        "Skin health score is an overall indicator combining hydration, texture, "
        "redness, and other facial skin parameters.",


    # ---------------- SCORE ----------------
    "scores.wellnessScore":
        "This score is based on the assessments you've completed. "
        "Offering a clear snapshot of your current health and fitness. "
        "Use it to track progress and target areas for improvement."
}


# ==========================================================
# ✅ SECTION TITLE
# ==========================================================
def section_title(text, styles):
    return Paragraph(
        text,
        ParagraphStyle(
            "section",
            parent=styles["Heading2"],
            alignment=1,
            backColor=colors.whitesmoke,
            spaceAfter=12,
            spaceBefore=18
        )
    )


# ==========================================================
# ✅ METRIC CARD BLOCK
# ==========================================================
def metric_block(title, metrics, path):

    value = get_metric(metrics, path, "value")
    unit = get_metric(metrics, path, "unit")
    status = get_metric(metrics, path, "interpretation")

    description = DESCRIPTIONS.get(path, "No description available.")

    table = Table(
        [
            [title, f"{value} {unit}"],
            ["Status", status]
        ],
        colWidths=[260, 180]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(38/255, 43/255, 97/255)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),

            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ])
    )

    return KeepTogether([
        Paragraph(f"<b>{title}</b>", getSampleStyleSheet()["Heading3"]),
        Spacer(1, 4),
        Paragraph(description, getSampleStyleSheet()["Normal"]),
        Spacer(1, 6),
        table,
        Spacer(1, 15)
    ])


# ==========================================================
# ✅ HRV DETAILS TABLE (Special Case)
# ==========================================================
def hrv_details_block(metrics):

    details = metrics["heartHealth"]["hrvDetails"]["value"]

    table = Table(
        [
            ["HRV Metric", "Value", "Unit"],
            ["SDNN", details["sdnn"], "ms"],
            ["RMSSD", details["rmssd"], "ms"],
            ["PNN50", details["pnn50"], "%"],
        ],
        colWidths=[200, 120, 120]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    )

    return KeepTogether([
        Paragraph("<b>HRV Details</b>", getSampleStyleSheet()["Heading3"]),
        Spacer(1, 6),
        table,
        Spacer(1, 15)
    ])


# ==========================================================
# ✅ MAIN PDF REPORT FUNCTION (ALL METRICS)
# ==========================================================
def generate_health_report(user, metrics, filename):

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # ==========================================================
    # HEADER
    # ==========================================================
    story.append(Paragraph("Assessment Report", styles["Title"]))
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            f"<b>Name :</b> {user['name']} &nbsp;&nbsp;&nbsp;"
            f"<b>Gender :</b> {user['gender']}<br/>"
            f"<b>Date of assessment :</b> {datetime.now().strftime('%d %b %Y')} "
            f"&nbsp;&nbsp;&nbsp;<b>Age :</b> {user['age']}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 25))

    # ==========================================================
    # ALL SECTIONS + ALL METRICS
    # ==========================================================
    sections = {
        "Overall Health Score": [
            ("Wellness Score", "scores.wellnessScore"),
        ],

        "Key Body Vitals": [
            ("Heart Rate", "vitals.heartRate"),
            ("Respiration Rate", "vitals.respiration"),
            ("Oxygen Saturation (SpO2)", "vitals.spo2"),
        ],

        "Heart Health": [
            ("HRV", "heartHealth.hrv"),
            ("Pulse Regularity", "heartHealth.pulseRegularity"),
            ("Cardiac Workload", "heartHealth.cardiacWorkload"),
            ("Recovery Index", "heartHealth.recoveryIndex"),
        ],

        "Mental Wellness": [
            ("Stress Level", "mental.stress"),
            ("Fatigue Score", "mental.fatigue"),
            ("Relaxation Score", "mental.relaxationScore"),
        ],

        "Behavior Metrics": [
            ("Blink Rate", "behavior.blinkRate"),
            ("Eye Closure Duration", "behavior.eyeClosureDuration"),
            ("Motion Stability", "behavior.motionStability"),
            ("Alertness", "behavior.alertness"),
        ],
         "Skin Health": [
            ("Skin Redness", "skin.skinRedness"),
            ("Skin Texture", "skin.skinTexture"),
            ("Skin Hydration", "skin.skinHydration"),
            ("Dark Circles", "skin.darkCircles"),
            ("Skin Health Score", "skin.skinHealthScore"),
        ]
    }

    # ==========================================================
    # PRINT EVERYTHING AUTOMATICALLY
    # ==========================================================
    for section_name, metrics_list in sections.items():

        story.append(section_title(section_name, styles))

        for title, path in metrics_list:
            story.append(metric_block(title, metrics, path))

        # Add HRV Details Table after Heart Health
        if section_name == "Heart Health":
            story.append(hrv_details_block(metrics))

    # ==========================================================
    # RECOMMENDATION (Same PDF Text)
    # ==========================================================
    story.append(section_title("Recommendation", styles))

    story.append(
        Paragraph(
            """
            <b>Morning Routine</b><br/>
            • Cleanse your face with a gentle cleanser to remove impurities.<br/>
            • Apply a hydrating toner to balance the skin’s pH.<br/>
            • Use an antioxidant serum to protect your skin from environmental damage.<br/>
            • Apply a moisturizer to hydrate and nourish your skin.<br/>
            • Finish with a broad-spectrum sunscreen.<br/><br/>

            <b>Evening Routine</b><br/>
            • Double cleanse to remove makeup, sunscreen, and daily grime.<br/>
            • Apply a hydrating toner.<br/>
            • Use a targeted treatment serum such as retinol or hyaluronic acid.<br/>
            • Apply a richer moisturizer overnight.<br/><br/>

            <b>Weekly Treatments</b><br/>
            • Exfoliate 1–2 times a week with a gentle exfoliant.<br/>
            • Apply a hydrating or brightening face mask once a week.<br/><br/>

            <b>Lifestyle Tips</b><br/>
            • Ensure adequate water intake throughout the day.<br/>
            • Eat a balanced diet rich in fruits and vegetables.<br/>
            • Aim for 7–9 hours of quality sleep per night.<br/>
            • Always wear sunscreen even on cloudy days.<br/>
            """,
            styles["Normal"]
        )
    )

    # ==========================================================
    # DISCLAIMER (Same PDF Text)
    # ==========================================================
    story.append(section_title("Disclaimer", styles))

    story.append(
        Paragraph(
            "This assessment is only indicative and not necessarily a direct representation "
            "of your risk. This report is not diagnostic. If you have concerns, please seek "
            "guidance from a medical professional who may conduct a physical examination "
            "and further diagnostic tests as required.",
            styles["Normal"]
        )
    )

    # ==========================================================
    # BUILD PDF
    # ==========================================================
    doc.build(story)

    return filename

def get_user_details(user_id):
    url = "https://api.raphacure.com/api/v1/user/user-details"
    
    headers = {
        "x-microservice-id": "RaphaCure_Microservice"
    }
    
    params = {
        "user_id": user_id
    }
    
    response = requests.get(url, headers=headers, params=params)


    
    # Raise exception for bad responses (4xx / 5xx)
    response.raise_for_status()
    
    return response.json()["data"]