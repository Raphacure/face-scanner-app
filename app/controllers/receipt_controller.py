from typing import Any, Dict, List

from app.services.receipt_prescription_classifier import (
    CLASSIFICATION_MARGIN,
    classify_url,
)

MAX_URLS = 25

# When we pick a side (not uncertain), map how far past the margin we are → display %.
# Raw r/(r+h) often stays ~50/50 even for clear prints; users expect a strong lean instead.
WIN_FLOOR = 88.0
WIN_CAP = 99.0
EDGE_SCALE = 0.22


def split_percentages(receipt_raw: float, script_raw: float, label: str) -> tuple[float, float]:
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
        t = min(1.0, edge / EDGE_SCALE)
        computer = min(WIN_CAP, WIN_FLOOR + (WIN_CAP - WIN_FLOOR) * t)
        return round(100.0 - computer, 2), round(computer, 2)

    edge = max(0.0, h - r - margin)
    t = min(1.0, edge / EDGE_SCALE)
    handwritten = min(WIN_CAP, WIN_FLOOR + (WIN_CAP - WIN_FLOOR) * t)
    return round(handwritten, 2), round(100.0 - handwritten, 2)


def public_row(url: str, row: Dict[str, Any]) -> Dict[str, Any]:
    r_raw = float(row.get("receipt_raw", 0.0))
    h_raw = float(row.get("script_raw", 0.0))
    label = row.get("classification")
    comp = row.get("completeness")
    fields = comp.get("fields") if isinstance(comp, dict) else None
    amount_present = (
        isinstance(fields, dict)
        and isinstance(fields.get("amount"), dict)
        and bool(fields["amount"].get("likely_present"))
    )
    consult_present = (
        isinstance(fields, dict)
        and isinstance(fields.get("consultation_type"), dict)
        and bool(fields["consultation_type"].get("likely_present"))
    )
    classification_reason = str(row.get("classification_reason") or "")

    if label == "handwritten_prescription":
        document_type = "handwritten"
        document_category = "prescription"
    elif label == "computer_generated_receipt":
        document_type = "computer_generated"
        # If promoted by report cues, keep it as report; else billing cues -> invoice.
        if classification_reason == "printed_report_cues":
            is_invoice = False
        else:
            is_invoice = amount_present or consult_present
        document_category = "invoice" if is_invoice else "report"
    else:
        document_type = "uncertain"
        document_category = "report"

    handwritten_pct, computer_pct = split_percentages(r_raw, h_raw, label)

    out: Dict[str, Any] = {
        "url": url,
        "document_type": document_type,
        "document_category": document_category,
        "handwritten_percent": handwritten_pct,
        "computer_generated_percent": computer_pct,
    }
    if isinstance(comp, dict):
        out["completeness_percent"] = float(comp.get("completeness_percent", 0.0))
        out["present_count"] = int(comp.get("present_count", 0))
        out["total_fields"] = int(comp.get("total_fields", 7))
        fields = comp.get("fields")
        if isinstance(fields, dict):
            out["fields"] = fields
    return out


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
            results.append(public_row(u, row))
        except Exception as e:
            results.append({"url": u, "error": str(e)})

    return {"status": "success", "results": results}
