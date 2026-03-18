from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import requests
from reportlab.platypus import Image    
import os
from typing import Optional

from reportlab.platypus import Flowable
from reportlab.graphics.shapes import Drawing, Rect

class RoundedImage(Flowable):
    def __init__(self, img_path, width, height, radius=15):
        super().__init__()
        self.img_path = img_path
        self.width = width
        self.height = height
        self.radius = radius

    def draw(self):
        c = self.canv

        # Create clipping path
        path = c.beginPath()
        path.roundRect(0, 0, self.width, self.height, self.radius)

        c.saveState()
        c.clipPath(path, stroke=0, fill=0)

        c.drawImage(
            self.img_path,
            0,
            0,
            width=self.width,
            height=self.height,
            mask='auto'
        )

        c.restoreState()
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


    "biological_age.biologicalAge":
        "Biological age represents how well your body is functioning compared to typical aging patterns."
        "It is estimated using physiological indicators such as cardiovascular health, stress, and recovery signals.",
       
    # ---------------- BEHAVIOR ----------------
    # "behavior.blinkRate":
    #     "Blink rate is the number of times you blink per minute. "
    #     "An increased blink rate may indicate stress, fatigue, or eye strain.",

    # "behavior.eyeClosureDuration":
    #     "Eye closure duration represents the average time your eyes remain closed. "
    #     "Longer closures may reflect tiredness or reduced alertness.",

    # "behavior.motionStability":
    #     "Motion stability measures how steady you remained during the scan. "
    #     "Higher stability improves scan accuracy.",

    # "behavior.alertness":
    #     "Alertness score reflects cognitive readiness and attentiveness. "
    #     "Higher alertness indicates better focus and awareness.",

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

    # ---------------- AYURVEDA DOSHA ----------------
    "dosha.vata":
        "Vata represents movement and communication in the body. "
        "It is associated with the nervous system, breathing, and circulation. "
        "Balanced Vata supports creativity, flexibility, and vitality.",

    "dosha.pitta":
        "Pitta represents metabolism and transformation. "
        "It governs digestion, body temperature, and energy production. "
        "Balanced Pitta supports focus, determination, and strong digestion.",

    "dosha.kapha":
        "Kapha represents structure and stability. "
        "It governs immunity, lubrication of joints, and emotional calmness. "
        "Balanced Kapha supports endurance, strength, and resilience.",


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
def generate_health_report(user, metrics, filename, image_path=None):

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # ==========================================================
    # TITLE
    # ==========================================================
    story.append(Paragraph("Assessment Report", styles["Title"]))
    story.append(Spacer(1, 25))

    # ----------------------------------------------------------
    # USER DETAILS
    # ----------------------------------------------------------
    user_details = Paragraph(
        f"""
        <b>Name :</b> {user['name']}<br/>
        <b>Gender :</b> {user['gender']}<br/>
        <b>Date of assessment :</b> {datetime.now().strftime('%d %b %Y')}<br/>
        <b>Age :</b> {user['age']}
        """,
        styles["Normal"]
    )

    # ----------------------------------------------------------
    # PROFILE IMAGE
    # ----------------------------------------------------------
    profile_image = ""

    if image_path and os.path.exists(image_path):
        try:
            profile_image = RoundedImage(image_path, width=140, height=110, radius=15)
        except:
            profile_image = ""

    # ----------------------------------------------------------
    # HEADER TABLE
    # ----------------------------------------------------------
    header_table = Table(
        [
            ["", user_details, profile_image]
        ],
        colWidths=[360, 200]  
    )
    header_table.hAlign = "RIGHT"

    header_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )

    story.append(header_table)
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

        # "Behavior Metrics": [
        #     ("Blink Rate", "behavior.blinkRate"),
        #     ("Eye Closure Duration", "behavior.eyeClosureDuration"),
        #     ("Motion Stability", "behavior.motionStability"),
        #     ("Alertness", "behavior.alertness"),
        # ],

        "Skin Health": [
            ("Skin Redness", "skin.skinRedness"),
            ("Skin Texture", "skin.skinTexture"),
            ("Skin Hydration", "skin.skinHydration"),
            ("Dark Circles", "skin.darkCircles"),
            ("Skin Health Score", "skin.skinHealthScore"),
        ],

        "Ayurveda Dosha Analysis": [
            ("Vata", "dosha.vata"),
            ("Pitta", "dosha.pitta"),
            ("Kapha", "dosha.kapha"),
        ],

        "Biological Age": [
            ("Biological Age", "biological_age.biologicalAge"),
        ],
    }

    # ==========================================================
    # PRINT METRICS
    # ==========================================================
    for section_name, metrics_list in sections.items():

        story.append(section_title(section_name, styles))

        for title, path in metrics_list:
            story.append(metric_block(title, metrics, path))

        if section_name == "Heart Health":
            story.append(hrv_details_block(metrics))

    # ==========================================================
    # RECOMMENDATION
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
    # DISCLAIMER
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


# ==========================================================
# ✅ SECOND PDF REPORT FUNCTION (LIGHTWEIGHT / TEMPLATE V2)
#    - Does NOT change the existing report.
#    - Produces a more compact, "under 2mb"-friendly report.
# ==========================================================
DEFAULT_SILHOUETTE_URL = (
    "https://raphacure-public-images.s3.ap-south-1.amazonaws.com/477456-1773644590391.png"
)


def _download_image_to_tmp(url: str, tmp_basename: str) -> Optional[str]:
    """
    Downloads an image to /tmp and returns the path.
    Returns None if download fails.
    """
    try:
        if not url:
            return None

        tmp_path = f"/tmp/{tmp_basename}"
        # If already downloaded in this worker, reuse it.
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            return tmp_path

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        return tmp_path
    except Exception:
        return None


def _safe_str(val, default="N/A"):
    if val is None:
        return default
    s = str(val).strip()
    return s if s else default


def _mini_metric_row(metrics, label, path, prefer_field="value"):
    val = get_metric(metrics, path, prefer_field, default="N/A")
    unit = get_metric(metrics, path, "unit", default="")
    interp = get_metric(metrics, path, "interpretation", default="N/A")
    value_text = f"{val} {unit}".strip()
    return [label, value_text, interp]


def _status_style(interp: str):
    s = (interp or "").strip().lower()
    if any(k in s for k in ["high", "poor", "irregular"]):
        return (colors.Color(0.97, 0.86, 0.86), colors.Color(0.75, 0.15, 0.15))  # red-ish
    if any(k in s for k in ["moderate", "medium"]):
        return (colors.Color(0.99, 0.93, 0.80), colors.Color(0.62, 0.45, 0.05))  # amber
    if "low" in s:
        return (colors.Color(0.92, 0.92, 0.92), colors.Color(0.35, 0.35, 0.35))  # neutral
    return (colors.Color(0.86, 0.96, 0.90), colors.Color(0.10, 0.55, 0.28))  # green-ish


def _metric_row_compact(styles, label: str, value: str, interp: str):
    pill_bg, pill_fg = _status_style(interp)
    label_style = ParagraphStyle(
        "v2_card_label",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.black,
        leading=8,
        wordWrap='LTR'
    )

    value_style = ParagraphStyle(
        "v2_card_value",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=colors.black,
        leading=9,
        alignment=2
    )
    pill_style = ParagraphStyle(
        "v2_card_pill",
        parent=styles["Normal"],
        fontSize=7,
        textColor=pill_fg,
        alignment=1,
        wordWrap="CJK",
    )

    # Status pill kept narrower and allows wrapping so the full row fits inside the card width
    pill = Table([[Paragraph(_safe_str(interp), pill_style)]], colWidths=[0.8 * inch])
    pill.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), pill_bg),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROUNDEDCORNERS", (0, 0), (-1, -1), 8),
            ]
        )
    )

    # Column widths sum is slightly less than the card width so content does not overflow the box
    row = Table(
        [[Paragraph(_safe_str(label), label_style), Paragraph(_safe_str(value), value_style), pill]],
        colWidths=[1.15 * inch, 0.70 * inch, 0.80 * inch]
    )
    row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return row

def _card(styles, title: str, rows):
    title_style = ParagraphStyle(
        "v2_card_title",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        textColor=colors.Color(0.16, 0.16, 0.16),
        spaceAfter=4,
        leading=9,
    )

    title_dot = Drawing(6,6)
    title_dot.add(Rect(0,0,6,6,fillColor=colors.Color(0.53,0.82,0.74),strokeColor=None))

    title_block = Table(
        [[title_dot, Paragraph(title.upper(), title_style)]],
        colWidths=[10, 160]
    )

    title_block.setStyle(TableStyle([
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("LEFTPADDING",(0,0),(-1,-1),0),
    ("RIGHTPADDING",(0,0),(-1,-1),4),
    ]))

    # Slightly narrower than the column in the outer body table to avoid visual overflow
    box = Table([[title_block], *[[r] for r in rows]], colWidths=[2.8 * inch])

    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.Color(0.90, 0.90, 0.90)),
                ("ROUNDEDCORNERS", (0, 0), (-1, -1), 8),

                # compact padding so content stays inside box
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    return box
class _ScoreRing(Flowable):
    def __init__(self, score: float, label: str = "Wellness Score"):
        super().__init__()
        self.score = score
        self.label = label
        self.width = 1.05 * inch
        self.height = 1.05 * inch

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height + 18)

    def draw(self):
        c = self.canv
        cx = self.width / 2
        cy = self.height / 2 + 10
        r = min(self.width, self.height) / 2 - 3

        c.saveState()
        c.setLineWidth(6)
        c.setStrokeColor(colors.Color(0.88, 0.95, 0.91))
        c.circle(cx, cy, r, stroke=1, fill=0)

        try:
            pct = max(0, min(100, float(self.score)))
        except Exception:
            pct = 0

        c.setStrokeColor(colors.Color(0.30, 0.76, 0.55))
        c.arc(cx - r, cy - r, cx + r, cy + r, startAng=90, extent=-360 * (pct / 100.0))

        c.setFillColor(colors.Color(0.16, 0.16, 0.16))
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(cx, cy + 2, f"{int(round(pct))}")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.grey)
        c.drawCentredString(cx, cy - 10, "/100")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.Color(0.25, 0.25, 0.25))
        c.drawCentredString(cx, 6, self.label)
        c.restoreState()


def _dosha_bar(percent: float, color_fill):
    # Width kept slightly less than the card width so the bar never overflows
    w = 2.4 * inch
    h = 7
    d = Drawing(w, h)
    d.add(Rect(0, 3, w, 4, fillColor=colors.Color(0.94, 0.94, 0.94), strokeColor=None))
    try:
        p = max(0, min(100, float(percent)))
    except Exception:
        p = 0
    d.add(Rect(0, 3, w * (p / 100.0), 4, fillColor=color_fill, strokeColor=None))
    return d


def generate_health_report_v2(
    user,
    metrics,
    filename,
    face_image_path=None,
    silhouette_url: str = DEFAULT_SILHOUETTE_URL,
):
    """
    Dashboard-style report matching the provided reference layout.
    """
    doc = SimpleDocTemplate(
        filename,
       pagesize=A4,
        leftMargin=18,
        rightMargin=18,
        topMargin=15,
        bottomMargin=15,
        title="Health Assessment Report",
        author="RaphaCure",
    )

    styles = getSampleStyleSheet()
    story = []

    tiny_style = ParagraphStyle(
        "v2_tiny",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.Color(0.40, 0.72, 0.64),
        leading=10,
    )
    h1_style = ParagraphStyle(
        "v2_h1",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.Color(0.15, 0.20, 0.30),
        alignment=0,  # left, like reference
        spaceAfter=0,
    )
    sub_style = ParagraphStyle(
        "v2_sub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        alignment=0,  # left
        spaceAfter=0,
        leading=11,
    )

    # Header: left side (title + name info) and right side (score ring) aligned on the same row
    name = _safe_str(user.get("name"), default="My Self")
    gender = _safe_str(user.get("gender"), default="N/A")
    age = _safe_str(user.get("age"), default="")
    date_txt = datetime.now().strftime("%b %-d , %Y") if os.name != "nt" else datetime.now().strftime("%b %d , %Y")

    # Left side: report title, name, and demographic line stacked vertically
    bio_age = get_metric(metrics, "biological_age.biologicalAge", "value", default="N/A")

    left_header_block = Table(
        [
            [Paragraph("HEALTH ASSESSMENT REPORT", tiny_style)],
            [Paragraph(name if name != "N/A" else "My Self", h1_style)],
            [Paragraph(f"{gender.title()} • Age {age} • {date_txt}".strip(" •"), sub_style)],
            [Paragraph(f"Biological Age {bio_age}".strip(), sub_style)],
        ],
        colWidths=[6.0 * inch],
    )

    wellness_value = get_metric(metrics, "scores.wellnessScore", "value", default=0)
    wellness_interp = get_metric(metrics, "scores.wellnessScore", "interpretation", default="")
    score_ring = _ScoreRing(wellness_value, label="")  # label handled separately

    score_status_style = ParagraphStyle(
        "v2_score_status",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.Color(0.30, 0.72, 0.64),
        alignment=1,
        spaceBefore=3,
    )
    score_caption_style = ParagraphStyle(
        "v2_score_caption",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=colors.Color(0.25, 0.27, 0.40),
        alignment=1,
        spaceBefore=0,
    )

    right_header_block = Table(
        [
            [score_ring],
            [Paragraph(_safe_str(wellness_interp), score_status_style)],
            [Paragraph("Wellness Score", score_caption_style)],
        ],
        colWidths=[1.2 * inch],
    )
    right_header_block.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    # Slightly narrower total width and a bit of right padding so the score block
    # does not stick to the page corner.
    header_row = Table([[left_header_block, right_header_block]], colWidths=[6.3 * inch, 1.3 * inch])
    header_row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(header_row)
    story.append(Spacer(1, 4))

    # Center silhouette
    silhouette_path = _download_image_to_tmp(silhouette_url, "report_v2_silhouette.png")
    silhouette_flowable = Spacer(1, 190)
    if silhouette_path and os.path.exists(silhouette_path):
        try:
            silhouette_flowable = Image(
                silhouette_path,
                width=1.25 * inch,
                height=2.6 * inch
            )
        except Exception:
            silhouette_flowable = Spacer(1, 210)

    # Left column
    mental_card = _card(
        styles,
        "MENTAL WELLNESS",
        [
            _metric_row_compact(
                styles,
                "Stress",
                f"{get_metric(metrics, 'mental.stress', 'value', default='--')} %",
                get_metric(metrics, "mental.stress", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Fatigue",
                f"{get_metric(metrics, 'mental.fatigue', 'value', default='--')} %",
                get_metric(metrics, "mental.fatigue", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Relaxation",
                f"{get_metric(metrics, 'mental.relaxationScore', 'value', default='--')} %",
                get_metric(metrics, "mental.relaxationScore", "interpretation", default="N/A"),
            ),
        ],
    )

    skin_card = _card(
        styles,
        "Skin Health",
        [
            _metric_row_compact(
                styles,
                "Skin Redness",
                f"{get_metric(metrics, 'skin.skinRedness', 'value', default='--')} %",
                get_metric(metrics, "skin.skinRedness", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Skin Texture",
                f"{get_metric(metrics, 'skin.skinTexture', 'value', default='--')} %",
                get_metric(metrics, "skin.skinTexture", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Skin Hydration",
                f"{get_metric(metrics, 'skin.skinHydration', 'value', default='--')} %",
                get_metric(metrics, "skin.skinHydration", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Dark Circles",
                f"{get_metric(metrics, 'skin.darkCircles', 'value', default='--')} %",
                get_metric(metrics, "skin.darkCircles", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Skin Health Score",
                f"{get_metric(metrics, 'skin.skinHealthScore', 'value', default='--')} %",
                get_metric(metrics, "skin.skinHealthScore", "interpretation", default="N/A"),
            ),
        ],
    )

    heart_card = _card(
        styles,
        "Heart Health",
        [
            _metric_row_compact(
                styles,
                "HRV",
                f"{get_metric(metrics, 'heartHealth.hrv', 'value', default='--')} ms",
                get_metric(metrics, "heartHealth.hrv", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Pulse Regularity",
                f"{get_metric(metrics, 'heartHealth.pulseRegularity', 'value', default='--')} %",
                get_metric(metrics, "heartHealth.pulseRegularity", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Cardiac Workload",
                f"{get_metric(metrics, 'heartHealth.cardiacWorkload', 'value', default='--')} score",
                get_metric(metrics, "heartHealth.cardiacWorkload", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Recovery Index",
                f"{get_metric(metrics, 'heartHealth.recoveryIndex', 'value', default='--')} %",
                get_metric(metrics, "heartHealth.recoveryIndex", "interpretation", default="N/A"),
            ),
        ],
    )
    # Avoid KeepTogether here; it can make the column "unsplittable" and crash if it doesn't fit.
    # Mental Wellness card will be shown on the right side with vitals/dosha
    left_col = [skin_card, Spacer(1, 6), heart_card]

    # Right column (no Behavior Metrics card in this layout)
    vitals_card = _card(
        styles,
        "Key Body Vitals",
        [
            _metric_row_compact(
                styles,
                "Heart Rate",
                f"{get_metric(metrics, 'vitals.heartRate', 'value', default='--')} bpm",
                get_metric(metrics, "vitals.heartRate", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Respiration Rate",
                f"{get_metric(metrics, 'vitals.respiration', 'value', default='--')} breaths/min",
                get_metric(metrics, "vitals.respiration", "interpretation", default="N/A"),
            ),
            _metric_row_compact(
                styles,
                "Oxygen Saturation",
                f"{get_metric(metrics, 'vitals.spo2', 'value', default='--')} %",
                get_metric(metrics, "vitals.spo2", "interpretation", default="N/A"),
            ),
        ],
    )

    vata = get_metric(metrics, "dosha.vata", "value", default=0)
    pitta = get_metric(metrics, "dosha.pitta", "value", default=0)
    kapha = get_metric(metrics, "dosha.kapha", "value", default=0)
    dosha_lbl = ParagraphStyle("v2_dosha_lbl", parent=styles["Normal"], fontSize=8.5, textColor=colors.Color(0.25, 0.25, 0.25))
    dosha_val = ParagraphStyle(
        "v2_dosha_val",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=colors.grey,
        alignment=2,
    )
    dosha_rows = []
    for name_lbl, pct, col in [
        ("Vata", vata, colors.Color(0.96, 0.78, 0.20)),
        ("Pitta", pitta, colors.Color(0.96, 0.55, 0.62)),
        ("Kapha", kapha, colors.Color(0.62, 0.78, 0.98)),
    ]:
        bar = _dosha_bar(pct, col)
        row = Table(
            [[Paragraph(name_lbl, dosha_lbl), bar, Paragraph(f"{_safe_str(pct)}% Normal", dosha_val)]],
            # Total width kept below card width (2.8") to avoid overflow
            colWidths=[0.55 * inch, 1.45 * inch, 0.7 * inch],
        )
        row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        dosha_rows.append(row)
    dosha_card = _card(styles, "AYURVEDA DOSHA", dosha_rows)

    # Right column: Key Body Vitals, Mental Wellness, then Ayurveda Dosha
    right_col = [vitals_card, Spacer(1, 6), mental_card, Spacer(1, 6), dosha_card]

    body = Table([[left_col, silhouette_flowable, right_col]], colWidths=[2.9 * inch, 1.5 * inch, 2.9 * inch])
    body.setStyle(
        TableStyle(
            [
                # Keep metric columns top-aligned
                ("VALIGN", (0, 0), (0, 0), "TOP"),
                ("VALIGN", (2, 0), (2, 0), "TOP"),
                # Center the human silhouette vertically in the middle column
                ("VALIGN", (1, 0), (1, 0), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(body)
    story.append(Spacer(1, 4))

    # HRV details strip
    # HRV DETAILS
    details = metrics.get("heartHealth", {}).get("hrvDetails", {}).get("value", {}) or {}

    sdnn = _safe_str(details.get("sdnn"), "--")
    rmssd = _safe_str(details.get("rmssd"), "--")
    pnn50 = _safe_str(details.get("pnn50"), "--")

    # Styles for HRV details block (matching pill-style dashboard look)
    hrv_title = Paragraph(
        "HRV DETAILS",
        ParagraphStyle(
            "hrv_title",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.Color(0.25, 0.25, 0.25),
            leading=11,
        ),
    )

    hrv_num = ParagraphStyle(
        "hrv_num",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.Color(0.10, 0.13, 0.24),
        alignment=1,
    )

    hrv_lbl = ParagraphStyle(
        "hrv_lbl",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        alignment=1,
    )

    def _hrv_bar(value_str, color):
        try:
            v = float(value_str)
        except Exception:
            v = 0
        # Normalise roughly into 0–100 range for visual only
        pct = max(0, min(100, v if "PNN" in "" else v / 10.0))
        w = 2.2 * inch
        h = 6
        d = Drawing(w, h)
        d.add(Rect(0, 1.5, w, 3, fillColor=colors.Color(0.93, 0.94, 0.96), strokeColor=None))
        d.add(Rect(0, 1.5, w * (pct / 100.0), 3, fillColor=color, strokeColor=None))
        return d

    hrv_block_style = TableStyle(
        [
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )

    sdnn_block = Table(
        [
            [Paragraph(sdnn, hrv_num)],
            [Paragraph("SDNN (ms)", hrv_lbl)],
            [_hrv_bar(sdnn, colors.Color(0.40, 0.78, 0.64))],
        ],
        colWidths=[2.4 * inch],
    )
    sdnn_block.setStyle(hrv_block_style)

    rmssd_block = Table(
        [
            [Paragraph(rmssd, hrv_num)],
            [Paragraph("RMSSD (ms)", hrv_lbl)],
            [_hrv_bar(rmssd, colors.Color(0.40, 0.64, 0.94))],
        ],
        colWidths=[2.4 * inch],
    )
    rmssd_block.setStyle(hrv_block_style)

    pnn50_block = Table(
        [
            [Paragraph(pnn50, hrv_num)],
            [Paragraph("PNN50 (%)", hrv_lbl)],
            [_hrv_bar(pnn50, colors.Color(0.96, 0.80, 0.32))],
        ],
        colWidths=[2.4 * inch],
    )
    pnn50_block.setStyle(hrv_block_style)

    hrv_table = Table(
        [[sdnn_block, rmssd_block, pnn50_block]],
        colWidths=[2.5 * inch, 2.5 * inch, 2.5 * inch],
    )
    hrv_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    hrv_container = Table(
        [
            [hrv_title],
            [hrv_table],
        ],
        colWidths=[7.6 * inch],
    )
    hrv_container.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, colors.Color(0.9, 0.9, 0.9)),
                ("ROUNDEDCORNERS", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )

    story.append(Spacer(1, 6))
    story.append(hrv_container)
    story.append(Spacer(1, 10))

    # Recommendations grid
    rec_txt = ParagraphStyle(
        "v2_rec_txt",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.Color(0.25, 0.27, 0.40),
        leading=12,
        wordWrap="LTR",
    )
    rec_title_style = ParagraphStyle(
        "v2_rec_title",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        textColor=colors.Color(0.18, 0.20, 0.35),
        leading=12,
        spaceAfter=2,
    )

    # Section heading "RECOMMENDATIONS"
    rec_section_title = Paragraph(
        "RECOMMENDATIONS",
        ParagraphStyle(
            "v2_rec_section_title",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=colors.Color(0.18, 0.20, 0.35),
            leading=14,
            spaceAfter=8,
        ),
    )
    story.append(rec_section_title)

    def _title_with_icon(icon_color, title_text):
        """
        Small colored dot + bold title text, side by side.
        This mimics the sun/moon/calendar/water icons in a minimal way
        that always renders with built‑in fonts.
        """
        # circle "icon"
        dot = Drawing(10, 10)
        dot.add(Rect(0, 3, 8, 8, fillColor=icon_color, strokeColor=icon_color))

        title_para = Paragraph(title_text, rec_title_style)
        title_row = Table([[dot, title_para]], colWidths=[10, 3.7 * inch - 10])
        title_row.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        return title_row

    def rec_card(icon_color, title_text, html):
        """
        Rounded white recommendation card with icon, title, and bulleted text,
        visually similar to the reference design.
        """
        title_row = _title_with_icon(icon_color, title_text)
        body_para = Paragraph(html, rec_txt)

        card = Table(
            [
                [title_row],
                [body_para],
            ],
            colWidths=[3.7 * inch],
        )
        # Draw a soft card with slightly rounded corners (not fully pill-shaped)
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.Color(0.93, 0.95, 0.99)),
                    # radius 10px-ish: clearly not rectangular, but still subtle
                    ("ROUNDEDCORNERS", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        return card

    rec_grid = Table(
        [
            [
                rec_card(
                    colors.Color(0.99, 0.73, 0.29),  # warm yellow
                    "Morning Routine",
                    "• Cleanse your face with a gentle cleanser<br/>"
                    "• Apply a hydrating toner to balance pH<br/>"
                    "• Use an antioxidant serum for protection<br/>"
                    "• Apply moisturizer to hydrate and nourish<br/>"
                    "• Finish with broad-spectrum sunscreen",
                ),
                rec_card(
                    colors.Color(0.35, 0.43, 0.94),  # cool blue
                    "Evening Routine",
                    "• Double cleanse to remove makeup and grime<br/>"
                    "• Apply a hydrating toner<br/>"
                    "• Use treatment serum (retinol or hyaluronic acid)<br/>"
                    "• Apply a richer moisturizer overnight",
                ),
            ],
            [
                rec_card(
                    colors.Color(0.96, 0.56, 0.56),  # soft red
                    "Weekly Treatments",
                    "• Exfoliate 1–2 times a week with a gentle exfoliant<br/>"
                    "• Apply a hydrating or brightening face mask",
                ),
                rec_card(
                    colors.Color(0.19, 0.84, 0.65),  # teal/green
                    "Lifestyle Tips",
                    "• Ensure adequate water intake throughout the day<br/>"
                    "• Eat a balanced diet rich in fruits and vegetables<br/>"
                    "• Aim for 7–9 hours of quality sleep<br/>"
                    "• Always wear sunscreen even on cloudy days",
                ),
            ],
        ],
        colWidths=[3.85 * inch, 3.85 * inch],
    )
    rec_grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(rec_grid)
    story.append(Spacer(1, 4))

    disc_style = ParagraphStyle("v2_disc", parent=styles["Normal"], fontSize=7.5, textColor=colors.grey, leading=10)
    story.append(
        Paragraph(
            "Disclaimer: This assessment is only indicative and not necessarily a direct representation of your risk. "
            "This report is not diagnostic. If you have concerns, please consult a medical professional.",
            disc_style,
        )
    )

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


def insert_face_scan(scanData):
    try:
        url = "https://api.raphacure.com/api/v1/face-scan"

        headers = {
            "x-microservice-id": "RaphaCure_Microservice",
            "Content-Type": "application/json"
        }

        response = requests.post(
            url,
            headers=headers,
            json=scanData,      # ✅ CORRECT
            timeout=10          # ✅ Always good practice
        )


        response.raise_for_status()

        return {
            "status": "success",
        }

    except requests.exceptions.RequestException as error:
        print("Face Scan API Error:", error)

        return {
            "status": "error",
            "message": "Face scan API request failed"
        }
