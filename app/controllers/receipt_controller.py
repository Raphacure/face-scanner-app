from typing import Any, Dict, List

from app.services.receipt_prescription_classifier import (
    CLASSIFICATION_MARGIN,
    classify_url,
)

MAX_URLS = 25

# When we pick a side (not uncertain), map how far past the margin we are → display %.
# Raw r/(r+h) often stays ~50/50 even for clear prints; users expect a strong lean instead.
_WIN_FLOOR = 88.0
_WIN_CAP = 99.0
_EDGE_SCALE = 0.22


def _split_percentages(receipt_raw: float, script_raw: float, label: str) -> tuple[float, float]:
    """Return (handwritten_percent, computer_generated_percent), sum ≈ 100."""
    r, h = receipt_raw, script_raw
    denom = r + h + 1e-9
    margin = CLASSIFICATION_MARGIN

    if label == "uncertain":
        hp = h / denom * 100.0
        cp = r / denom * 100.0
        return round(hp, 2), round(cp, 2)

    if label == "computer_generated_receipt":
        edge = max(0.0, r - h - margin)
        t = min(1.0, edge / _EDGE_SCALE)
        computer = min(_WIN_CAP, _WIN_FLOOR + (_WIN_CAP - _WIN_FLOOR) * t)
        return round(100.0 - computer, 2), round(computer, 2)

    edge = max(0.0, h - r - margin)
    t = min(1.0, edge / _EDGE_SCALE)
    handwritten = min(_WIN_CAP, _WIN_FLOOR + (_WIN_CAP - _WIN_FLOOR) * t)
    return round(handwritten, 2), round(100.0 - handwritten, 2)


def _public_row(url: str, row: Dict[str, Any]) -> Dict[str, Any]:
    r_raw = float(row.get("receipt_raw", 0.0))
    h_raw = float(row.get("script_raw", 0.0))
    label = row.get("classification")
    if label == "computer_generated_receipt":
        doc_type = "computer_generated"
    elif label == "handwritten_prescription":
        doc_type = "handwritten"
    else:
        doc_type = "uncertain"

    handwritten_pct, computer_pct = _split_percentages(r_raw, h_raw, label)

    return {
        "url": url,
        "document_type": doc_type,
        "handwritten_percent": handwritten_pct,
        "computer_generated_percent": computer_pct,
    }


def classify_receipt_prescription_urls_controller(urls: List[str]) -> Dict[str, Any]:
    if not urls:
        return {"status": "error", "message": "urls must be a non-empty array"}
    if len(urls) > MAX_URLS:
        return {
            "status": "error",
            "message": f"At most {MAX_URLS} URLs allowed",
        }

    results: List[Dict[str, Any]] = []

    for url in urls:
        u = (url or "").strip()
        if not u:
            results.append({"url": url or "", "error": "empty url"})
            continue
        try:
            row = classify_url(u)
            results.append(_public_row(u, row))
        except Exception as e:
            results.append({"url": u, "error": str(e)})

    return {"status": "success", "results": results}
