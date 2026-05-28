"""
Heuristic classifier: computer-generated receipt vs handwritten prescription.

Uses OpenCV-only signals (layout regularity, horizontal text bands, contour
irregularity). Results are best-effort; ambiguous cases return "uncertain".
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
import requests

from app.services.prescription_completeness import analyze_prescription_completeness

MAX_IMAGE_BYTES = 10 * 1024 * 1024
REQUEST_TIMEOUT_S = 20
MAX_SIDE = 900

# Same margin for label and for mapping raw score gap → display percentages.
CLASSIFICATION_MARGIN = 0.12


def fetch_image_bgr(url: str) -> np.ndarray:
    headers = {"User-Agent": "face-ai-service/receipt-classify"}
    with requests.get(
        url,
        timeout=REQUEST_TIMEOUT_S,
        stream=True,
        headers=headers,
    ) as r:
        r.raise_for_status()
        data = bytearray()
        for chunk in r.iter_content(65536):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds size limit")
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def resize_long_side(gray: np.ndarray, max_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def row_projection_peaks(proj: np.ndarray, min_dist: int) -> tuple[int, float]:
    p = proj.astype(np.float64)
    if p.max() <= 0:
        return 0, 0.0
    p = p / p.max()
    min_h = 0.12
    peaks: list[int] = []
    for i in range(1, len(p) - 1):
        if p[i] >= min_h and p[i] >= p[i - 1] and p[i] >= p[i + 1]:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
    if len(peaks) < 3:
        return len(peaks), 0.0
    gaps = np.diff(np.array(peaks, dtype=np.float64))
    reg = float(np.std(gaps) / (np.mean(gaps) + 1e-6))
    return len(peaks), reg


def raw_receipt_script_scores(gray: np.ndarray) -> tuple[float, float]:
    gray = resize_long_side(gray, MAX_SIDE)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, gray.shape[1] // 25), 1))
    horiz_strokes = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)
    ink = float(bw.sum()) + 1.0
    horiz_ratio = float(horiz_strokes.sum()) / ink

    proj = bw.sum(axis=1)
    min_dist = max(4, gray.shape[0] // 100)
    n_peaks, gap_cv = row_projection_peaks(proj, min_dist)

    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(30, gray.shape[1] // 25),
        minLineLength=max(25, gray.shape[1] // 8),
        maxLineGap=12,
    )
    horiz_lines = 0
    total_lines = 0
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            total_lines += 1
            dy, dx = abs(y2 - y1), abs(x2 - x1)
            if dx > 20 and dy <= max(3, gray.shape[0] // 80):
                horiz_lines += 1
    horiz_line_frac = (horiz_lines / total_lines) if total_lines else 0.0

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [float(cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 15.0]
    areas.sort(reverse=True)
    top = areas[: min(120, len(areas))]
    if len(top) < 8:
        area_irregularity = 0.35
    else:
        area_irregularity = float(np.std(top) / (np.mean(top) + 1e-6))

    receipt_raw = (
        1.4 * horiz_ratio
        + 0.55 * min(1.0, n_peaks / 35.0)
        + 0.9 * horiz_line_frac
        + 0.35 * min(1.0, 1.0 / (gap_cv + 0.25))
    )
    script_raw = 0.9 * area_irregularity + 0.45 * min(1.0, gap_cv / 1.2) + 0.25 * (1.0 - horiz_ratio)

    return float(receipt_raw), float(script_raw)


def _is_likely_present(completeness: Dict[str, Any], key: str, min_conf: float = 0.0) -> bool:
    fields = completeness.get("fields")
    if not isinstance(fields, dict):
        return False
    entry = fields.get(key)
    if not isinstance(entry, dict):
        return False
    if not bool(entry.get("likely_present")):
        return False
    return float(entry.get("confidence_percent", 0.0)) >= min_conf


def refine_label_with_completeness(label: str, completeness: Dict[str, Any]) -> tuple[str, str]:
    """
    Correct obvious printed invoices that raw texture scoring can mark handwritten.
    """
    amount_present = _is_likely_present(completeness, "amount", min_conf=55.0)
    consult_present = _is_likely_present(completeness, "consultation_type", min_conf=55.0)
    hospital_name_present = _is_likely_present(completeness, "hospital_name", min_conf=45.0)
    hospital_address_present = _is_likely_present(completeness, "hospital_address", min_conf=45.0)
    doctor_present = _is_likely_present(completeness, "doctor_name", min_conf=40.0)

    printed_invoice_cues = amount_present and consult_present and (
        hospital_name_present or hospital_address_present
    )
    if printed_invoice_cues:
        return "computer_generated_receipt", "printed_invoice_cues"

    printed_report_cues = (not amount_present) and consult_present and (
        (hospital_name_present and hospital_address_present) or doctor_present
    )
    if printed_report_cues and label != "computer_generated_receipt":
        return "computer_generated_receipt", "printed_report_cues"
    return label, "raw_score"


def classify_url(url: str) -> Dict[str, Any]:
    img = fetch_image_bgr(url)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, h = raw_receipt_script_scores(gray)
    m = CLASSIFICATION_MARGIN
    if r > h + m:
        label = "computer_generated_receipt"
    elif h > r + m:
        label = "handwritten_prescription"
    else:
        label = "uncertain"

    denom = r + h + 1e-6
    is_printed_receipt = label == "computer_generated_receipt"
    # Handwritten cash-memo CV; printed bills use column/total digit detection instead.
    allow_amount_cv = h >= r and not is_printed_receipt
    completeness = analyze_prescription_completeness(
        img,
        gray,
        allow_amount_cv=allow_amount_cv,
        is_printed_receipt=is_printed_receipt,
    )

    refined_label, refine_reason = refine_label_with_completeness(label, completeness)
    if refined_label != label:
        label = refined_label
        is_printed_receipt = label == "computer_generated_receipt"
        allow_amount_cv = h >= r and not is_printed_receipt
        completeness = analyze_prescription_completeness(
            img,
            gray,
            allow_amount_cv=allow_amount_cv,
            is_printed_receipt=is_printed_receipt,
        )

    return {
        "url": url,
        "classification": label,
        "receipt_raw": float(r),
        "script_raw": float(h),
        "scores": {
            "receipt": float(r / denom),
            "prescription": float(h / denom),
        },
        "classification_reason": refine_reason,
        "completeness": completeness,
    }

