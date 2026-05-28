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
STAMP_PRESENT_THRESHOLD = 0.45

STAMP_KW = (
    "stamp",
    "seal",
    "authorised",
    "authorized",
    "registered",
    "rubber stamp",
)

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
    "op consultation",
    "op consult",
    "op cons",
    "consultation",
    "particulars",
    "follow-up",
    "follow up",
    "followup",
    "visit type",
    "new patient",
    "review",
)
OP_CONSULT_RE = re.compile(r"op\s*cons", re.IGNORECASE)
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


def _colored_stamp_mask(hsv: np.ndarray) -> np.ndarray:
    """Hue filter tolerant of faded blue/purple ink after JPEG compression."""
    red_lo = cv2.inRange(hsv, (0, 40, 40), (14, 255, 255))
    red_hi = cv2.inRange(hsv, (165, 40, 40), (180, 255, 255))
    blue = cv2.inRange(hsv, (88, 14, 32), (148, 255, 255))
    purple = cv2.inRange(hsv, (100, 14, 32), (178, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red_lo, red_hi), cv2.bitwise_or(blue, purple))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return cv2.dilate(mask, k, iterations=1)


def _stamp_blob_score(mask: np.ndarray, img_area: float) -> float:
    """Score round-ish ink blobs; rings often fragment so circularity is relaxed."""
    best = 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < img_area * 0.0015 or area > img_area * 0.18:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1e-3:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / float(ch + 1e-6)
        if ar < 0.45 or ar > 2.2:
            continue
        hull = cv2.convexHull(c)
        solidity = area / (float(cv2.contourArea(hull)) + 1e-6)
        round_w = min(1.0, circularity / 0.28)
        ar_w = 1.0 - min(1.0, abs(ar - 1.0) / 0.75)
        size_w = min(1.0, area / (img_area * 0.008))
        solid_w = min(1.0, solidity / 0.55)
        best = max(best, 0.35 * round_w + 0.25 * ar_w + 0.25 * size_w + 0.15 * solid_w)
    return float(min(1.0, best))


def _stamp_region_score(bgr: np.ndarray) -> float:
    h, w = bgr.shape[:2]
    if h < 30 or w < 30:
        return 0.0
    img_area = float(h * w)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    color_mask = _colored_stamp_mask(hsv)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (41, 41), 0)
    diff = cv2.absdiff(bg, gray)
    _, diff_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    ink_mask = cv2.bitwise_and(diff_mask, color_mask)

    return max(_stamp_blob_score(color_mask, img_area), _stamp_blob_score(ink_mask, img_area))


def stamp_likelihood(bgr: np.ndarray, *, text: str = "") -> float:
    small = resize_bgr(bgr, MAX_SIDE)
    h, w = small.shape[:2]
    if h < 40 or w < 40:
        return 0.0

    # Rubber stamps usually sit in the footer; scan a few bands and take the best.
    regions = (
        (0.72, 1.0, 0.0, 0.58),  # bottom-left (common on receipts)
        (0.48, 1.0, 0.0, 1.0),  # lower half fallback
        (0.68, 0.96, 0.42, 1.0),  # bottom-right signature area
    )
    best = 0.0
    for y0f, y1f, x0f, x1f in regions:
        roi = small[int(h * y0f) : int(h * y1f), int(w * x0f) : int(w * x1f)]
        best = max(best, _stamp_region_score(roi))

    kw = keyword_score(text, STAMP_KW)
    if kw >= 0.5:
        best = max(best, 0.72)
    return float(min(1.0, best))


def _footer_signature_score(foot: np.ndarray) -> float:
    """Freehand strokes in a footer crop; penalize all-horizontal printed lines."""
    fh, fw = foot.shape[:2]
    if fh < 12 or fw < 20:
        return 0.0

    edges = cv2.Canny(foot, 45, 130)
    edge_density = float(np.count_nonzero(edges)) / float(fh * fw + 1e-6)
    if edge_density > 0.11:
        return float(max(0.0, 0.25 - (edge_density - 0.11) * 2.5))

    bw = cv2.adaptiveThreshold(
        foot, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    ink = float(bw.sum()) + 1.0
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, fw // 18), 1))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    horiz_ratio = float(hor.sum()) / ink
    if horiz_ratio > 0.85:
        return 0.0
    stroke = min(1.0, edge_density * 12.0)
    freehand = 0.25 + 0.75 * (1.0 - min(1.0, horiz_ratio * 1.2))
    return float(stroke * freehand)


def _stamp_area_signature_score(bgr: np.ndarray) -> float:
    """Handwritten signature often sits inside the rubber stamp (bottom-left)."""
    small = resize_bgr(bgr, MAX_SIDE)
    h, w = small.shape[:2]
    roi = small[int(h * 0.70) :, : int(w * 0.50)]
    rh, rw = roi.shape[:2]
    if rh < 20 or rw < 20:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_mask = _colored_stamp_mask(hsv)
    if np.count_nonzero(color_mask) < rh * rw * 0.001:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 110)
    sig_edges = cv2.bitwise_and(edges, color_mask)
    color_area = float(np.count_nonzero(color_mask)) + 1e-6
    stroke_density = float(np.count_nonzero(sig_edges)) / color_area
    return float(min(1.0, stroke_density * 2.8))


def signature_likelihood(
    gray: np.ndarray,
    bgr: np.ndarray | None = None,
    *,
    stamp_score: float = 0.0,
) -> float:
    small = resize_gray(gray, MAX_SIDE)
    h, w = small.shape[:2]

    br = _footer_signature_score(small[int(h * 0.82) : h, int(w * 0.52) : w])
    bl = 0.0
    if bgr is not None:
        bl = _stamp_area_signature_score(bgr)
    if stamp_score >= STAMP_PRESENT_THRESHOLD:
        bl = max(bl, min(0.88, stamp_score * 0.92))

    return float(min(1.0, max(br, bl)))


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


def particulars_table_score(gray: np.ndarray) -> float:
    """Line-item band on printed bills (e.g. OP CONSULTATION row)."""
    small = resize_gray(gray, MAX_SIDE)
    h, w = small.shape[:2]
    band = small[int(h * 0.36) : int(h * 0.62), :]
    if band.size == 0:
        return 0.0
    bw = cv2.adaptiveThreshold(
        band, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8
    )
    proj = bw.sum(axis=1).astype(np.float64)
    if proj.max() <= 0:
        return 0.0
    proj = proj / proj.max()
    peaks = sum(
        1
        for i in range(1, len(proj) - 1)
        if proj[i] >= 0.28 and proj[i] >= proj[i - 1] and proj[i] >= proj[i + 1]
    )
    if peaks >= 3:
        return 0.78
    if peaks >= 1:
        return 0.42
    return 0.0


def consultation_type_score(text: str, gray: np.ndarray) -> float:
    if OP_CONSULT_RE.search(text):
        return 0.92
    kw = keyword_score(text, CONSULT_KW)
    table = particulars_table_score(gray)
    body = band_ink_score(gray, 0.35, 0.78)
    layout = body * 0.5 if kw < 0.35 else kw
    return float(min(1.0, max(kw, table, layout)))


def _digit_cluster_score(roi: np.ndarray) -> float:
    """Count printed currency-style digit blobs in a ROI."""
    rh, rw = roi.shape[:2]
    if rh < 15 or rw < 15:
        return 0.0
    bw = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9
    )
    roi_area = float(rh * rw)
    digitish = 0
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < roi_area * 0.00025:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        wf, hf = cw / float(rw), ch / float(rh)
        if 0.008 <= wf <= 0.28 and 0.012 <= hf <= 0.14:
            digitish += 1
    if digitish >= 4:
        return 0.88
    if digitish >= 2:
        return 0.72
    if digitish >= 1:
        return 0.45
    return 0.0


def printed_receipt_amount_score(gray: np.ndarray) -> float:
    """Amount columns / totals on structured printed bills (no OCR required)."""
    small = resize_gray(gray, MAX_SIDE)
    h, w = small.shape[:2]
    if h < 80 or w < 80:
        return 0.0

    regions = (
        (0.35, 0.72, 0.48, 0.98),  # line-item amount column
        (0.50, 0.82, 0.52, 0.98),  # gross / net summary
        (0.78, 0.94, 0.35, 0.98),  # payment row
    )
    best = 0.0
    for y0f, y1f, x0f, x1f in regions:
        roi = small[int(h * y0f) : int(h * y1f), int(w * x0f) : int(w * x1f)]
        best = max(best, _digit_cluster_score(roi))
    return float(best)


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


def amount_score(
    text: str,
    gray: np.ndarray,
    *,
    allow_cv: bool = True,
    is_printed_receipt: bool = False,
) -> float:
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
        if re.search(r"\b\d{2,5}\.\d{2}\b", text):
            return 0.75
    if is_printed_receipt:
        printed = printed_receipt_amount_score(gray)
        if printed >= PRESENT_THRESHOLD:
            return printed
    if not allow_cv:
        return 0.0
    return max(amount_cv_score(gray), printed_receipt_amount_score(gray) * 0.85)


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
    is_printed_receipt: bool = False,
) -> Dict[str, Any]:
    """
    Return per-field cues and completeness_percent (equal weight per field).

    allow_amount_cv: use layout-based amount detection (for handwritten bills).
    Set False for printed clinical summaries to avoid false positives.
    is_printed_receipt: enable printed-bill amount columns / totals detection.
    """
    text = ocr_text(gray)
    stamp_score = stamp_likelihood(bgr, text=text)

    scores: Dict[str, float] = {
        "stamp": stamp_score,
        "hospital_name": hospital_name_score(text, gray),
        "hospital_address": hospital_address_score(text, gray),
        "doctor_name": doctor_name_score(text, gray),
        "doctor_signature": signature_likelihood(
            gray, bgr, stamp_score=stamp_score
        ),
        "consultation_type": consultation_type_score(text, gray),
        "amount": amount_score(
            text,
            gray,
            allow_cv=allow_amount_cv,
            is_printed_receipt=is_printed_receipt,
        ),
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
