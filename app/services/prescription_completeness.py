"""
Heuristic prescription / receipt completeness from an image.

Scores seven fields (stamp, hospital name, address, doctor name, signature,
consultation type, amount). Each present field contributes equally to
completeness_percent. Uses OCR when pytesseract is available; otherwise
layout and keyword fallbacks on band crops.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

MAX_SIDE = 900

FIELD_KEYS = (
    "stamp",
    "hospital_name",
    "hospital_address",
    "doctor_name",
    "doctor_signature",
    "consultation_type",
    "amount",
)

PRESENT_THRESHOLD = 0.38
STAMP_PRESENT_THRESHOLD = 0.55

HOSPITAL_KW = (
    "hospital",
    "clinic",
    "medical",
    "healthcare",
    "health care",
    "nursing home",
    "polyclinic",
    "diagnostic",
    "centre",
    "center",
    "institute",
)
DOCTOR_KW = ("dr.", "dr ", "doctor", "physician", "mbbs", "md ", "m.d", "consultant")
CONSULT_KW = (
    "opd",
    "ipd",
    "consultation",
    "follow-up",
    "follow up",
    "followup",
    "visit type",
    "new patient",
    "review",
)
AMOUNT_KW = (
    "rs.",
    "rs ",
    "inr",
    "total",
    "amount",
    "fee",
    "fees",
    "charges",
    "charge",
    "rupees",
    "₹",
    "price",
    "paid",
    "payment",
)
PIN_RE = re.compile(r"\b\d{6}\b")
AMOUNT_NUM_RE = re.compile(
    r"(?:rs\.?|inr|₹|total|amount|fee|charges?)\s*[:\-]?\s*(\d[\d,]*\.?\d*)",
    re.IGNORECASE,
)
# Billing line only — avoids refraction / age numbers on clinical summaries.
BILLING_LINE_RE = re.compile(
    r"(?:^|\n)[^\n]{0,90}(?:rs\.?|inr|₹|total\s*amount|amount\s*due|"
    r"bill\s*amount|grand\s*total|fee|fees|charges?|payment|rupees)"
    r"[^\n]{0,40}\d",
    re.IGNORECASE,
)


def resize_bgr(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def resize_gray(gray: np.ndarray, max_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray
    scale = max_side / float(m)
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def ocr_text(gray: np.ndarray) -> str:
    try:
        import pytesseract  # type: ignore

        text = pytesseract.image_to_string(gray, config="--psm 6")
        return (text or "").lower()
    except Exception:
        return ""


def band_ink_score(gray: np.ndarray, y0_frac: float, y1_frac: float) -> float:
    """Ink density + row structure in a horizontal band (0..1)."""
    small = resize_gray(gray, MAX_SIDE)
    h, w = small.shape[:2]
    y0, y1 = int(h * y0_frac), int(h * y1_frac)
    band = small[y0:y1, :]
    if band.size == 0:
        return 0.0
    bw = cv2.adaptiveThreshold(
        band, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8
    )
    ink = float(np.count_nonzero(bw)) / float(bw.size + 1e-6)
    proj = bw.sum(axis=1).astype(np.float64)
    if proj.max() <= 0:
        return float(min(1.0, ink * 4.0))
    proj = proj / proj.max()
    peaks = sum(
        1
        for i in range(1, len(proj) - 1)
        if proj[i] >= 0.15 and proj[i] >= proj[i - 1] and proj[i] >= proj[i + 1]
    )
    row_score = min(1.0, peaks / 5.0)
    return float(min(1.0, 0.45 * min(1.0, ink * 5.5) + 0.55 * row_score))


def keyword_score(text: str, keywords: Tuple[str, ...]) -> float:
    if not text:
        return 0.0
    hits = sum(1 for k in keywords if k in text)
    return float(min(1.0, hits / max(1, min(2, len(keywords) // 4 + 1))))


def stamp_likelihood(bgr: np.ndarray) -> float:
    small = resize_bgr(bgr, MAX_SIDE)
    h, w = small.shape[:2]
    if h < 40 or w < 40:
        return 0.0

    # Stamps sit on the doctor/footer area, not letterhead logos.
    small = small[int(h * 0.48) :, :]
    h, w = small.shape[:2]
    if h < 30:
        return 0.0

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    red_lo = cv2.inRange(hsv, (0, 55, 55), (12, 255, 255))
    red_hi = cv2.inRange(hsv, (168, 55, 55), (180, 255, 255))
    blue = cv2.inRange(hsv, (95, 50, 50), (138, 255, 255))
    purple = cv2.inRange(hsv, (118, 35, 35), (168, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red_lo, red_hi), cv2.bitwise_or(blue, purple))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    img_area = float(h * w)
    best = 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < img_area * 0.001 or area > img_area * 0.22:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1e-3:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)
        if circularity < 0.42:
            continue
        shape_w = min(1.0, circularity / 0.72)
        size_w = min(1.0, area / (img_area * 0.012))
        best = max(best, 0.55 * shape_w + 0.45 * size_w)
    return float(min(1.0, best))


def signature_likelihood(gray: np.ndarray) -> float:
    """Bottom-right corner only; penalize dense printed tables in the footer."""
    small = resize_gray(gray, MAX_SIDE)
    h, w = small.shape[:2]
    y0, x0 = int(h * 0.82), int(w * 0.52)
    foot = small[y0:h, x0:w]
    fh, fw = foot.shape[:2]
    if fh < 12 or fw < 20:
        return 0.0

    edges = cv2.Canny(foot, 45, 130)
    edge_density = float(np.count_nonzero(edges)) / float(fh * fw + 1e-6)
    # Printed footers / tables have very high edge density — not a signature.
    if edge_density > 0.11:
        return float(max(0.0, 0.25 - (edge_density - 0.11) * 2.5))

    bw = cv2.adaptiveThreshold(
        foot, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    ink = float(bw.sum()) + 1.0
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, fw // 18), 1))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    horiz_ratio = float(hor.sum()) / ink
    if horiz_ratio > 0.5:
        return 0.0
    stroke = min(1.0, edge_density * 9.0)
    freehand = 0.25 + 0.75 * (1.0 - min(1.0, horiz_ratio * 1.2))
    return float(stroke * freehand)


def hospital_name_score(text: str, gray: np.ndarray) -> float:
    kw = keyword_score(text, HOSPITAL_KW)
    header = band_ink_score(gray, 0.0, 0.32)
    return float(min(1.0, max(kw, header * 0.72 if kw < 0.35 else kw)))


def hospital_address_score(text: str, gray: np.ndarray) -> float:
    if PIN_RE.search(text):
        return 0.92
    addr_kw = ("road", "street", "nagar", "lane", "avenue", "sector", "block", "floor")
    kw = keyword_score(text, addr_kw)
    mid = band_ink_score(gray, 0.12, 0.48)
    digit_lines = 0.0
    if re.search(r"\d{2,}", text):
        digit_lines = 0.55
    return float(min(1.0, max(kw, digit_lines, mid * 0.65)))


def doctor_name_score(text: str, gray: np.ndarray) -> float:
    kw = keyword_score(text, DOCTOR_KW)
    upper_mid = band_ink_score(gray, 0.18, 0.55)
    return float(min(1.0, max(kw, upper_mid * 0.58 if kw < 0.4 else kw)))


def consultation_type_score(text: str, gray: np.ndarray) -> float:
    kw = keyword_score(text, CONSULT_KW)
    body = band_ink_score(gray, 0.35, 0.78)
    return float(min(1.0, max(kw, body * 0.5 if kw < 0.35 else kw)))


def amount_cv_score(gray: np.ndarray) -> float:
    """
    Detect handwritten billing amounts in the right column / total row (e.g. 4200/-).
    Skips dense printed clinical grids (OPD summaries).
    """
    small = resize_gray(gray, MAX_SIDE)
    h, w = small.shape[:2]
    if h < 80 or w < 80:
        return 0.0

    full_bw = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11
    )
    proj = full_bw.sum(axis=1).astype(np.float64)
    row_peaks = 0
    if proj.max() > 0:
        p = proj / proj.max()
        for i in range(1, len(p) - 1):
            if p[i] >= 0.14 and p[i] >= p[i - 1] and p[i] >= p[i + 1]:
                row_peaks += 1
    # Printed OPD / refraction grids — many uniform rows, no cash-memo style total.
    if row_peaks >= 26:
        return 0.0

    def roi_score(y0f: float, y1f: float, x0f: float, x1f: float) -> float:
        roi = small[int(h * y0f) : int(h * y1f), int(w * x0f) : int(w * x1f)]
        rh, rw = roi.shape[:2]
        if rh < 20 or rw < 20:
            return 0.0
        bw = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9
        )
        roi_area = float(rh * rw)
        best = 0.0
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = float(cv2.contourArea(c))
            if area < roi_area * 0.0025:
                continue
            wf, hf = cw / float(rw), ch / float(rh)
            # Handwritten amount / total: wide stroke group, not a tiny printed cell.
            if wf >= 0.14 and 0.02 <= hf <= 0.2 and area >= roi_area * 0.004:
                blob = min(1.0, wf / 0.32) * min(1.0, area / (roi_area * 0.02))
                best = max(best, blob)
        return float(best)

    # Amount column (right) + total row (bottom)
    col = roi_score(0.32, 0.88, 0.55, 0.98)
    total = roi_score(0.72, 0.97, 0.38, 0.98)
    return float(min(1.0, max(col, total) * 1.05))


def amount_score(text: str, gray: np.ndarray, *, allow_cv: bool = True) -> float:
    """
    Billing amount: OCR when available; CV fallback on receipts / cash memos only.
    """
    if text.strip():
        if AMOUNT_NUM_RE.search(text) or BILLING_LINE_RE.search(text):
            return 0.92
        kw = keyword_score(text, AMOUNT_KW)
        if kw >= 0.5 and re.search(
            r"(?:rs\.?|inr|₹|total|amount|fee|charges?|payment).{0,35}\d",
            text,
            re.IGNORECASE,
        ):
            return 0.78
    if not allow_cv:
        return 0.0
    return amount_cv_score(gray)


def field_entry(score: float, *, threshold: float = PRESENT_THRESHOLD) -> Dict[str, Any]:
    present = score >= threshold
    return {
        "likely_present": present,
        "confidence_percent": round(score * 100.0, 2),
    }


def analyze_prescription_completeness(
    bgr: np.ndarray,
    gray: np.ndarray,
    *,
    allow_amount_cv: bool = True,
) -> Dict[str, Any]:
    """
    Return per-field cues and completeness_percent (equal weight per field).

    allow_amount_cv: use layout-based amount detection (for handwritten bills).
    Set False for printed clinical summaries to avoid false positives.
    """
    text = ocr_text(gray)

    scores: Dict[str, float] = {
        "stamp": stamp_likelihood(bgr),
        "hospital_name": hospital_name_score(text, gray),
        "hospital_address": hospital_address_score(text, gray),
        "doctor_name": doctor_name_score(text, gray),
        "doctor_signature": signature_likelihood(gray),
        "consultation_type": consultation_type_score(text, gray),
        "amount": amount_score(text, gray, allow_cv=allow_amount_cv),
    }

    n = len(FIELD_KEYS)
    weight = 100.0 / n
    fields: Dict[str, Any] = {}
    present_count = 0
    completeness = 0.0

    for key in FIELD_KEYS:
        th = STAMP_PRESENT_THRESHOLD if key == "stamp" else PRESENT_THRESHOLD
        entry = field_entry(scores[key], threshold=th)
        if entry["likely_present"]:
            present_count += 1
            completeness += weight
        entry["contribution_percent"] = round(weight if entry["likely_present"] else 0.0, 2)
        fields[key] = entry

    return {
        "completeness_percent": round(completeness, 2),
        "present_count": present_count,
        "total_fields": n,
        "fields": fields,
        "ocr_available": bool(text.strip()),
    }
