"""
AWS Textract OCR helpers for claim documents.

Used with OpenAI Vision: Textract fills printed fields (GSTIN, invoice no,
amounts, drug license); OpenAI classifies document type and medical fields.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.aws.textract_client import (
    disable_textract_runtime,
    get_textract_client,
    textract_enabled,
)

logger = logging.getLogger(__name__)

# Sync Bytes API soft limit (AWS: 5 MB for most sync ops).
_MAX_SYNC_BYTES = 4 * 1024 * 1024

_GSTIN_RE = re.compile(
    r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b",
    re.IGNORECASE,
)
_GSTIN_PREFIX_RE = re.compile(
    r"(?<![A-Z0-9])(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]{2}[A-Z0-9])(?![A-Z0-9])",
    re.IGNORECASE,
)

_DL_STATE_FORMAT_RE = re.compile(
    r"(\d{1,2}-DRUG/\d{4}-\d{2,4}/\d+)",
    re.IGNORECASE,
)
_DL_CHAIN_FORMAT_RE = re.compile(
    r"(?:DL\s*NO\.?\s*(?:\d+)?\s*[:\s#-]*)?(\d{1,2}-\d{5,7})",
    re.IGNORECASE,
)
_DL_NUMERIC_PAIR_RE = re.compile(r"\b(\d{1,2}-\d{5,7})\b")
_DL_SLASH_FORMAT_RE = re.compile(
    r"(\d{1,2}/MD/[A-Z]{2,5}/\d+)",
    re.IGNORECASE,
)
_DL_LICENCE_SLASH_RE = re.compile(
    r"(\d{2,3}/\d{4,6}/\d{2}/[A-Z]+-\d{2,3})",
    re.IGNORECASE,
)

_INVOICE_NO_RE = re.compile(
    r"(?:invoice\s*(?:no|number|#)|inv\.?\s*no\.?|bill\s*no\.?|receipt\s*no\.?)"
    r"\s*[:#.\-]?\s*([A-Za-z0-9][A-Za-z0-9/\-]{1,24})",
    re.IGNORECASE,
)
_TOTAL_RE = re.compile(
    r"(?:grand\s*total|total\s*(?:amount|mrp\s*value|invoice\s*value)?|net\s*amount|"
    r"amount\s*payable|bill\s*amount)\s*[:\-]?\s*(?:rs\.?|₹|inr)?\s*"
    r"([\d,]+\.?\d*)",
    re.IGNORECASE,
)


def _extract_gstin(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[*#]+", " ", text.upper())
    compact = re.sub(r"\s+", "", cleaned)
    compact = re.sub(r"GSTIN(?:/UIN)?[:/]?|GST\s*NO[./:]?", "|", compact)
    compact = re.sub(r"\|+", "|", compact)
    compact = re.sub(r"FSSAI(?:\s*NO)?[:/]?\d{8,}", "|", compact)
    match = _GSTIN_RE.search(compact)
    if match:
        return match.group(1).upper()
    labelled = re.search(
        r"(?:GSTIN(?:/UIN)?|GSTNO)[:/|]*([A-Z0-9]{15})",
        compact,
    )
    if labelled:
        cand = labelled.group(1).upper()
        if _GSTIN_RE.fullmatch(cand):
            return cand
        if len(cand) == 15 and cand[13] != "Z":
            fixed = cand[:13] + "Z" + cand[14]
            if _GSTIN_RE.fullmatch(fixed):
                return fixed
    for cand_match in _GSTIN_PREFIX_RE.finditer(compact):
        candidate = cand_match.group(1).upper()
        if len(candidate) != 15:
            continue
        if candidate[13] != "Z":
            fixed = candidate[:13] + "Z" + candidate[14:]
            if _GSTIN_RE.fullmatch(fixed):
                return fixed
    return ""


def _is_plausible_drug_license(lic: str) -> bool:
    if (
        _DL_STATE_FORMAT_RE.fullmatch(lic)
        or _DL_SLASH_FORMAT_RE.fullmatch(lic)
        or _DL_LICENCE_SLASH_RE.fullmatch(lic)
    ):
        return True
    match = re.fullmatch(r"(\d{1,2})-(\d{5,7})", lic)
    if not match:
        return False
    return int(match.group(1)) >= 10


def _extract_drug_licenses(text: str) -> List[str]:
    if not text:
        return []
    seen: List[str] = []
    for pattern in (
        _DL_STATE_FORMAT_RE,
        _DL_CHAIN_FORMAT_RE,
        _DL_SLASH_FORMAT_RE,
        _DL_LICENCE_SLASH_RE,
    ):
        for match in pattern.finditer(text):
            lic = match.group(1)
            if _is_plausible_drug_license(lic) and lic not in seen:
                seen.append(lic)
    numeric_matches = [m.group(1) for m in _DL_NUMERIC_PAIR_RE.finditer(text)]
    if numeric_matches and (
        len(numeric_matches) >= 2
        or re.search(r"\bDL\b|\bdrug\s*lic", text, re.IGNORECASE)
    ):
        for lic in numeric_matches:
            if _is_plausible_drug_license(lic) and lic not in seen:
                seen.append(lic)
    return seen


def _lines_from_detect(blocks: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for block in blocks or []:
        if block.get("BlockType") == "LINE" and block.get("Text"):
            lines.append(str(block["Text"]).strip())
    return [line for line in lines if line]


def _expense_field_map(response: Dict[str, Any]) -> Dict[str, str]:
    """Map AnalyzeExpense summary types → values."""
    out: Dict[str, str] = {}
    for doc in response.get("ExpenseDocuments") or []:
        for field in doc.get("SummaryFields") or []:
            ftype = ((field.get("Type") or {}).get("Text") or "").strip().upper()
            value = ((field.get("ValueDetection") or {}).get("Text") or "").strip()
            if ftype and value and ftype not in out:
                out[ftype] = value
    return out


def _normalize_amount(raw: str) -> str:
    if not raw:
        return ""
    clean = re.sub(r"[^\d.]", "", raw.replace(",", ""))
    if not clean:
        return ""
    if clean.count(".") > 1:
        parts = clean.split(".")
        clean = "".join(parts[:-1]) + "." + parts[-1]
    return clean


def _pick_bytes_for_sync(
    document_raw: bytes,
    page_images: Optional[List[bytes]] = None,
) -> bytes:
    """Prefer first rendered page if raw PDF/image is too large for sync Bytes API."""
    if page_images:
        page = page_images[0]
        if page and len(page) <= _MAX_SYNC_BYTES:
            return page
    if document_raw and len(document_raw) <= _MAX_SYNC_BYTES:
        return document_raw
    if page_images and page_images[0]:
        return page_images[0]
    return document_raw


_EMPTY_OCR: Dict[str, str] = {
    "gst_number": "",
    "drug_license_number": "",
    "invoice_number": "",
    "invoice_date": "",
    "total_amount": "",
    "provider_name": "",
    "patient_name": "",
    "provider_address": "",
}


def extract_textract_fields(
    document_raw: bytes,
    page_images: Optional[List[bytes]] = None,
) -> Dict[str, str]:
    """Run Textract OCR; return claim fields (empty strings when unknown)."""
    if not textract_enabled() or not document_raw:
        return dict(_EMPTY_OCR)

    payload = _pick_bytes_for_sync(document_raw, page_images)
    if not payload:
        return dict(_EMPTY_OCR)

    client = get_textract_client()
    lines: List[str] = []
    expense: Dict[str, str] = {}

    try:
        detect = client.detect_document_text(Document={"Bytes": payload})
        lines = _lines_from_detect(detect.get("Blocks") or [])
    except Exception as exc:
        _log_textract_error("detect_document_text", exc)

    if not textract_enabled():
        return dict(_EMPTY_OCR)

    try:
        expense_resp = client.analyze_expense(Document={"Bytes": payload})
        expense = _expense_field_map(expense_resp)
    except Exception as exc:
        _log_textract_error("analyze_expense", exc)

    text = "\n".join(lines)
    gst = _extract_gstin(text)
    if not gst:
        for val in expense.values():
            gst = _extract_gstin(val) or gst
            if gst:
                break

    licenses = _extract_drug_licenses(text)
    invoice_number = (expense.get("INVOICE_RECEIPT_ID") or "").strip()
    if not invoice_number:
        inv_match = _INVOICE_NO_RE.search(text)
        if inv_match:
            invoice_number = inv_match.group(1).strip()

    total = _normalize_amount(expense.get("TOTAL") or "")
    if not total:
        tot_match = _TOTAL_RE.search(text)
        if tot_match:
            total = _normalize_amount(tot_match.group(1))

    provider = (expense.get("VENDOR_NAME") or "").strip()
    patient = (
        expense.get("NAME")
        or expense.get("CUSTOMER_NAME")
        or expense.get("RECEIVER_NAME")
        or ""
    ).strip()
    if patient and provider and patient.lower() == provider.lower():
        patient = ""

    return {
        "gst_number": gst,
        "drug_license_number": "; ".join(licenses) if licenses else "",
        "invoice_number": invoice_number,
        "invoice_date": (expense.get("INVOICE_RECEIPT_DATE") or "").strip(),
        "total_amount": total,
        "provider_name": provider,
        "patient_name": patient,
        "provider_address": (expense.get("VENDOR_ADDRESS") or "").strip(),
    }


def _log_textract_error(operation: str, exc: Exception) -> None:
    """AccessDenied: warn once and disable Textract for this process."""
    name = type(exc).__name__
    if "AccessDenied" in name or "AccessDenied" in str(exc):
        disable_textract_runtime()
        logger.warning(
            "Textract AccessDenied (need textract:DetectDocumentText + "
            "textract:AnalyzeExpense on IAM). OpenAI-only until policy is fixed. (%s)",
            operation,
        )
        return
    logger.exception("Textract %s failed", operation)


def merge_textract_into_openai_data(
    data: Dict[str, Any],
    ocr: Dict[str, str],
) -> None:
    """Fill empty OpenAI invoice / top-level gaps with Textract OCR values."""
    if not ocr:
        return

    inv_raw = data.get("invoice_parameters")
    inv: Dict[str, Any] = dict(inv_raw) if isinstance(inv_raw, dict) else {}

    def _blank(val: Any) -> bool:
        return not str(val or "").strip()

    fill_keys = (
        "gst_number",
        "drug_license_number",
        "invoice_number",
        "invoice_date",
        "total_amount",
        "provider_name",
        "patient_name",
        "provider_address",
    )
    for key in fill_keys:
        ocr_val = (ocr.get(key) or "").strip()
        if ocr_val and _blank(inv.get(key)):
            inv[key] = ocr_val

    # Prefer valid GSTIN from OCR even when OpenAI returned garbage.
    ocr_gst = (ocr.get("gst_number") or "").strip()
    if ocr_gst and _extract_gstin(ocr_gst):
        existing = _extract_gstin(str(inv.get("gst_number") or ""))
        if not existing:
            inv["gst_number"] = ocr_gst

    ocr_dl = (ocr.get("drug_license_number") or "").strip()
    if ocr_dl and _blank(inv.get("drug_license_number")):
        inv["drug_license_number"] = ocr_dl

    if inv:
        data["invoice_parameters"] = inv

    # If OpenAI missed category but expense OCR looks like a bill, nudge recovery.
    if not data.get("is_medical_document", True) and (
        ocr_gst or (ocr.get("invoice_number") and ocr.get("total_amount"))
    ):
        data["is_medical_document"] = True
        if str(data.get("document_category", "other")) in ("other", ""):
            data["document_category"] = "invoice"
