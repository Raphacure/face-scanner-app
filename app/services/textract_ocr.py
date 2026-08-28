"""
AWS Textract OCR helpers for claim documents.

Used with OpenAI Vision: Textract fills printed fields (GSTIN, invoice no,
amounts, drug license, payment txn ids); OpenAI classifies document type
and medical / payment fields.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence
import os

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
_AGE_SEX_RE = re.compile(
    r"((?:\d{1,3})\s*(?:years?|yrs?|y)?(?:\s*\d+\s*months?)?)\s*[/\-,]?\s*"
    r"(male|female|m|f)\b",
    re.IGNORECASE,
)
_DOCTOR_LINE_RE = re.compile(
    r"(?:^|\n)\s*Doctor\s*[:\-]?\s*(Dr\.?\s*[A-Za-z][A-Za-z.\s]{2,60})",
    re.IGNORECASE,
)
_FACILITY_LINE_RE = re.compile(
    r"(?:^|\n)\s*Facility\s*[:\-]?\s*([^\n]{3,80})",
    re.IGNORECASE,
)
_APPT_DATE_RE = re.compile(
    r"(?:Appt\.?\s*Dt|Note\s*Dt|Visit\s*Date|Date)\s*[:\-]?\s*"
    r"([0-9]{1,2}[\s/|.\-][A-Za-z]{3,9}'?\s*\d{2,4}|[0-9]{1,2}[/|.\-][0-9]{1,2}[/|.\-][0-9]{2,4})",
    re.IGNORECASE,
)
_STANDALONE_RX_DATE_RE = re.compile(
    r"(?:^|\n)\s*(\d{1,2}[/|.\-]\d{1,2}[/|.\-]\d{2,4})\s*(?:\n|$)",
    re.MULTILINE,
)
_SYSTEMIC_RE = re.compile(
    r"Systemic\s*History\s*:?\s*(.+?)(?:\n\s*Allergies|\n\s*GLASSES|\n\s*REFRACTION|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_IOP_RE = re.compile(
    r"IOP\s*\(?\d*\)?\s*:?\s*(\d+)\s*(?:at\s*[^\n]*)?",
    re.IGNORECASE,
)
_VA_RE = re.compile(r"\bVA\s*:?\s*([^\n]{3,40})", re.IGNORECASE)

# --- Payment receipt / UPI / bank transfer cues ---
# Require PSP-like handles; reject emails (info@hospital / name@domain.com).
_UPI_ID_RE = re.compile(
    r"\b([a-zA-Z0-9][a-zA-Z0-9._-]{1,64}@[a-zA-Z][a-zA-Z0-9]{1,20})\b"
)
_KNOWN_UPI_HANDLES = frozenset(
    {
        "upi",
        "ybl",
        "ibl",
        "axl",
        "apl",
        "paytm",
        "okhdfcbank",
        "oksbi",
        "okicici",
        "okaxis",
        "okbizaxis",
        "waaxis",
        "wahdfcbank",
        "ptyes",
        "ptaxis",
        "yesbank",
        "freecharge",
        "amazonpay",
        "ikwik",
        "jupiteraxis",
        "indus",
        "kbl",
        "barodampay",
        "uboi",
        "cbin",
        "idbi",
    }
)
_UTR_RE = re.compile(
    r"(?:UTR(?:\s*(?:No\.?|Number|#))?|UPI\s*Ref(?:erence)?(?:\s*No\.?)?|"
    r"Bank\s*Ref(?:erence)?(?:\s*No\.?)?)\s*[:#\-]?\s*([A-Z0-9]{8,30})",
    re.IGNORECASE,
)
_TXN_ID_RE = re.compile(
    r"(?:Txn(?:saction)?\s*(?:ID|Id|No\.?|#)|Transaction\s*(?:ID|Id|No\.?|#)|"
    r"Payment\s*ID|RRN)\s*[:#\-]?\s*([A-Z0-9]{8,30})",
    re.IGNORECASE,
)
_PAYMENT_AMOUNT_RE = re.compile(
    r"(?:(?:paid|sent|transferred)\s+(?:of\s+)?)?"
    r"(?:₹|rs\.?|inr)\s*([\d,]+\.?\d{0,2})"
    r"|(?:(?:paid|amount\s*paid|total\s*paid)\s*[:\-]?\s*(?:₹|rs\.?|inr)?\s*"
    r"([\d,]+\.?\d{0,2}))",
    re.IGNORECASE,
)
_PAYMENT_DATE_RE = re.compile(
    r"(?:(?:paid\s*on|transaction\s*date|payment\s*date)\s*[:\-]?\s*)"
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"
    r"|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_PAYMENT_TIME_RE = re.compile(
    r"\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b"
)
_IFSC_RE = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", re.IGNORECASE)
_MASKED_ACCT_RE = re.compile(
    r"(?:A/?C|Account|Acc(?:ount)?\s*(?:No\.?)?)\s*[:#\-]?\s*"
    r"([Xx*•·.\-\s]*\d{2,6})",
    re.IGNORECASE,
)
_PAYEE_RE = re.compile(
    r"(?:(?:paid|sent|transferred)\s+to|payee|merchant|beneficiary)"
    r"\s*[:\-]?\s*([A-Za-z][A-Za-z0-9 .,&'\-]{2,60})",
    re.IGNORECASE,
)
_PAYER_RE = re.compile(
    r"(?:paid\s+by|debited\s+from|payer)\s*[:\-]?\s*"
    r"([A-Za-z][A-Za-z0-9 .,&'\-]{2,60})",
    re.IGNORECASE,
)
# Require "Bank" for short brands (Axis alone is a refraction column header).
_BANK_NAME_RE = re.compile(
    r"\b((?:HDFC|ICICI|SBI|State Bank of India|Kotak|IDFC|PNB|"
    r"Bank of Baroda|Canara|IndusInd|IDBI|YES)\s+Bank"
    r"|Axis\s+Bank|Yes\s+Bank|Union\s+Bank(?:\s+of\s+India)?|"
    r"Federal\s+Bank)\b",
    re.IGNORECASE,
)
_PAYMENT_APP_RE = re.compile(
    r"\b(Google\s*Pay|GPay|PhonePe|Paytm|BHIM|Amazon\s*Pay|CRED|Mobikwik)\b",
    re.IGNORECASE,
)
_PAYMENT_STATUS_RE = re.compile(
    r"\b(payment\s+successful|transaction\s+successful|successfully\s+paid|"
    r"payment\s+failed|transaction\s+failed|payment\s+pending|"
    r"payment\s+successful|transaction\s+failed)\b",
    re.IGNORECASE,
)
_PAYMENT_MODE_SIGNAL_RE = re.compile(
    r"\b(UPI|NEFT|IMPS|RTGS|RTPS|credit\s*card|debit\s*card|net\s*banking|"
    r"wallet)\b",
    re.IGNORECASE,
)
_MEDICAL_DOC_CUE_RE = re.compile(
    r"\b(?:OPD\s*SUMMARY|prescription|refraction|visual\s*acuity|\bIOP\b|"
    r"chief\s*complaints|systemic\s*history|glasses\s*prescription|"
    r"auto\s*refraction|pathologist|diagnosis|Dr\.?\s)\b",
    re.IGNORECASE,
)


def _is_plausible_upi_id(value: str) -> bool:
    """Accept real UPI VPAs; reject emails / hospital footer addresses."""
    raw = (value or "").strip()
    if "@" not in raw or " " in raw:
        return False
    local, _, handle = raw.partition("@")
    if not local or not handle:
        return False
    handle_l = handle.lower()
    local_l = local.lower()
    # Emails / domains
    if "." in handle_l or handle_l.endswith(("com", "in", "org", "net", "co")):
        return False
    if local_l in {"info", "admin", "support", "contact", "hello", "mail", "email"}:
        return False
    if handle_l in _KNOWN_UPI_HANDLES:
        return True
    # Common PSP-style prefixes (ptaxis, okhdfcbank, …)
    if re.match(r"^(ok|pt|yb|ibl|axl|apl|wa)", handle_l):
        return True
    # Unknown short handle only if local part looks like a user id (has digit)
    if len(handle_l) <= 12 and any(ch.isdigit() for ch in local):
        return True
    return False


def _extract_gstin(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[*#]+", " ", text.upper())
    compact = re.sub(r"\s+", "", cleaned)
    compact = re.sub(r"GSTIN(?:/UIN)?[:/]?|GST\s*NO[./:]?", "|", compact)
    compact = re.sub(r"\|+", "|", compact)
    compact = re.sub(r"FSSAI(?:\s*NO)?[:/]?\d{8,}", "|", compact)

    def _normalize_candidate(cand: str) -> str:
        cand = cand.upper()
        if len(cand) != 15:
            return ""
        chars = list(cand)
        # Position 13 must be Z; OCR often misreads the digit before Z as I/l/O.
        if chars[13] != "Z":
            chars[13] = "Z"
        if chars[12] in ("I", "L"):
            chars[12] = "1"
        elif chars[12] == "O":
            chars[12] = "0"
        fixed = "".join(chars)
        if _GSTIN_RE.fullmatch(fixed):
            return fixed
        return cand if _GSTIN_RE.fullmatch(cand) else ""

    match = _GSTIN_RE.search(compact)
    if match:
        return _normalize_candidate(match.group(1)) or match.group(1).upper()
    labelled = re.search(
        r"(?:GSTIN(?:/UIN)?|GSTNO)[:/|]*([A-Z0-9]{15})",
        compact,
    )
    if labelled:
        repaired = _normalize_candidate(labelled.group(1))
        if repaired:
            return repaired
    for cand_match in _GSTIN_PREFIX_RE.finditer(compact):
        repaired = _normalize_candidate(cand_match.group(1))
        if repaired:
            return repaired
    return ""


def _value_after_label(lines: List[str], labels: Sequence[str]) -> str:
    """Return the next non-empty line after a label line (or same-line value)."""
    label_set = {lab.lower() for lab in labels}
    for idx, line in enumerate(lines):
        raw = line.strip()
        low = raw.lower().rstrip(":")
        if low in label_set:
            if idx + 1 < len(lines):
                nxt = lines[idx + 1].strip()
                if nxt and nxt.lower().rstrip(":") not in label_set:
                    return nxt
            continue
        for lab in labels:
            prefix = lab.lower() + ":"
            if low.startswith(prefix):
                return raw.split(":", 1)[1].strip()
    return ""


def _parse_demographics_from_lines(lines: List[str]) -> Dict[str, str]:
    text = "\n".join(lines)
    out: Dict[str, str] = {
        "patient_name": "",
        "patient_age": "",
        "patient_gender": "",
        "doctor_name": "",
        "clinic_hospital_name": "",
        "consultation_date": "",
        "diagnosis": "",
        "visual_acuity_details": "",
        "provider_contact": "",
    }

    age_sex = _AGE_SEX_RE.search(text)
    if age_sex:
        out["patient_age"] = age_sex.group(1).strip()
        gender = age_sex.group(2).strip().upper()
        out["patient_gender"] = (
            "F" if gender.startswith("F") else "M" if gender.startswith("M") else gender
        )

    patient = _value_after_label(lines, ["Patient", "Patient Name", "Patient's Name"])
    if patient:
        out["patient_name"] = patient

    doctor = _value_after_label(lines, ["Doctor", "Consultant", "Unit Consultants"])
    if not doctor:
        dm = _DOCTOR_LINE_RE.search(text)
        doctor = dm.group(1).strip() if dm else ""
    if doctor and len(doctor) > 3:
        out["doctor_name"] = doctor

    facility = _value_after_label(lines, ["Facility", "Hospital", "Clinic"])
    if not facility:
        fm = _FACILITY_LINE_RE.search(text)
        facility = fm.group(1).strip() if fm else ""
    if facility and len(facility) > 3:
        out["clinic_hospital_name"] = facility

    dm = _APPT_DATE_RE.search(text)
    if dm:
        out["consultation_date"] = re.sub(r"\s+", " ", dm.group(1)).strip().replace("|", "/")
    if not out["consultation_date"]:
        sm = _STANDALONE_RX_DATE_RE.search(text)
        if sm:
            out["consultation_date"] = sm.group(1).strip().replace("|", "/")

    systemic = _SYSTEMIC_RE.search(text)
    if systemic:
        diag = re.sub(r"\s+", " ", systemic.group(1)).strip(" :-\n")
        if diag and diag.lower() not in {"none", "nil", "n/a", "-"}:
            out["diagnosis"] = diag

    vas = [m.group(0).strip() for m in _VA_RE.finditer(text)]
    iops = [f"IOP {m.group(1)}" for m in _IOP_RE.finditer(text)]
    va_bits = []
    if vas:
        # de-dupe preserve order
        seen: set[str] = set()
        for v in vas:
            key = v.lower()
            if key not in seen:
                seen.add(key)
                va_bits.append(v)
    if iops:
        va_bits.extend(dict.fromkeys(iops))
    if va_bits:
        out["visual_acuity_details"] = "; ".join(va_bits)

    contact = _value_after_label(lines, ["Contact", "Mobile", "Mobile No", "Phone"])
    if contact and re.search(r"\d{8,}", contact):
        out["provider_contact"] = contact

    return out


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


def _infer_payment_mode(text: str) -> str:
    low = (text or "").lower()
    if re.search(r"\b(upi|gpay|google\s*pay|phonepe|paytm|bhim|amazon\s*pay)\b", low):
        return "upi"
    if re.search(r"\b(neft|imps|rtgs|rtps|bank\s*transfer|net\s*banking)\b", low):
        return "bank_transfer"
    if re.search(r"\b(credit\s*card|debit\s*card|visa|mastercard|rupay|card)\b", low):
        return "card"
    if re.search(r"\bwallet\b", low):
        return "wallet"
    if re.search(r"\bcash\b", low):
        return "cash"
    mode = _PAYMENT_MODE_SIGNAL_RE.search(text or "")
    if not mode:
        return ""
    token = mode.group(1).lower()
    if token == "upi":
        return "upi"
    if token in ("neft", "imps", "rtgs", "rtps"):
        return "bank_transfer"
    if "card" in token:
        return "card"
    if token == "wallet":
        return "wallet"
    if token == "cash":
        return "cash"
    if "net" in token:
        return "bank_transfer"
    return ""


def _infer_payment_status(text: str) -> str:
    match = _PAYMENT_STATUS_RE.search(text or "")
    if not match:
        return ""
    raw = match.group(1).lower()
    if "fail" in raw or "declin" in raw:
        return "failed"
    if "pend" in raw:
        return "pending"
    if "complete" in raw:
        return "completed"
    if "success" in raw:
        return "success"
    return ""


def _extract_payment_fields(text: str) -> Dict[str, str]:
    """Best-effort payment proof fields from OCR text (empty when unknown)."""
    out: Dict[str, str] = {
        "payment_mode": "",
        "payment_amount": "",
        "transaction_date": "",
        "transaction_id": "",
        "reference_number": "",
        "utr": "",
        "payer_name": "",
        "payee_name": "",
        "upi_id": "",
        "bank_name": "",
        "payment_status": "",
        "payment_time": "",
        "account_number_masked": "",
        "ifsc": "",
        "remarks": "",
    }
    if not text:
        return out

    # Clinical / OPD pages often contain Axis/email/time noise — skip payment OCR.
    if _MEDICAL_DOC_CUE_RE.search(text) and not (
        _PAYMENT_APP_RE.search(text)
        or re.search(r"\b(?:payment\s+successful|UTR|UPI\s*Ref|NEFT|IMPS)\b", text, re.I)
    ):
        return out

    amt = _PAYMENT_AMOUNT_RE.search(text)
    if amt:
        amount = _normalize_amount(amt.group(1) or amt.group(2) or "")
        try:
            if amount and float(amount) > 0:
                out["payment_amount"] = amount
        except ValueError:
            if amount:
                out["payment_amount"] = amount

    utr_m = _UTR_RE.search(text)
    if utr_m:
        out["utr"] = utr_m.group(1).strip()
        out["transaction_id"] = out["utr"]
        out["reference_number"] = out["utr"]

    txn_m = _TXN_ID_RE.search(text)
    if txn_m:
        txn = txn_m.group(1).strip()
        if not out["transaction_id"]:
            out["transaction_id"] = txn
        if not out["reference_number"]:
            out["reference_number"] = txn

    for upi_m in _UPI_ID_RE.finditer(text):
        candidate = upi_m.group(1).strip()
        if _is_plausible_upi_id(candidate):
            out["upi_id"] = candidate
            break

    date_m = _PAYMENT_DATE_RE.search(text)
    if date_m:
        out["transaction_date"] = re.sub(r"\s+", " ", date_m.group(1)).strip()

    ifsc_m = _IFSC_RE.search(text)
    if ifsc_m:
        out["ifsc"] = ifsc_m.group(1).upper()

    acct_m = _MASKED_ACCT_RE.search(text)
    if acct_m:
        out["account_number_masked"] = re.sub(r"\s+", "", acct_m.group(1)).strip()

    bank_m = _BANK_NAME_RE.search(text)
    if bank_m:
        out["bank_name"] = re.sub(r"\s+", " ", bank_m.group(1)).strip()

    payee_m = _PAYEE_RE.search(text)
    if payee_m:
        candidate = re.sub(r"\s+", " ", payee_m.group(1)).strip(" :-")
        if len(candidate) >= 3 and "@" not in candidate:
            out["payee_name"] = candidate[:80]

    payer_m = _PAYER_RE.search(text)
    if payer_m:
        candidate = re.sub(r"\s+", " ", payer_m.group(1)).strip(" :-")
        if len(candidate) >= 3 and "@" not in candidate:
            out["payer_name"] = candidate[:80]

    out["payment_mode"] = _infer_payment_mode(text)
    out["payment_status"] = _infer_payment_status(text)

    # App name alone is a UPI cue when mode still empty.
    if not out["payment_mode"] and _PAYMENT_APP_RE.search(text):
        out["payment_mode"] = "upi"

    # Time only when this already looks like a payment proof.
    if out["payment_status"] or out["utr"] or out["upi_id"] or out["payment_mode"]:
        time_m = _PAYMENT_TIME_RE.search(text)
        if time_m:
            out["payment_time"] = time_m.group(1).strip()

    return out


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
    "patient_age": "",
    "patient_gender": "",
    "doctor_name": "",
    "clinic_hospital_name": "",
    "consultation_date": "",
    "diagnosis": "",
    "visual_acuity_details": "",
    "provider_contact": "",
    "payment_mode": "",
    "payment_amount": "",
    "transaction_date": "",
    "transaction_id": "",
    "reference_number": "",
    "utr": "",
    "payer_name": "",
    "payee_name": "",
    "upi_id": "",
    "bank_name": "",
    "payment_status": "",
    "payment_time": "",
    "account_number_masked": "",
    "ifsc": "",
    "remarks": "",
    "raw_text": "",
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

    use_expense = (os.getenv("TEXTRACT_ANALYZE_EXPENSE") or "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    def _detect() -> List[str]:
        try:
            detect = client.detect_document_text(Document={"Bytes": payload})
            return _lines_from_detect(detect.get("Blocks") or [])
        except Exception as exc:
            _log_textract_error("detect_document_text", exc)
            return []

    def _expense() -> Dict[str, str]:
        if not use_expense:
            return {}
        try:
            expense_resp = client.analyze_expense(Document={"Bytes": payload})
            return _expense_field_map(expense_resp)
        except Exception as exc:
            _log_textract_error("analyze_expense", exc)
            return {}

    # Detect + AnalyzeExpense in parallel (~saves one serial Textract RTT).
    if use_expense:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_lines = pool.submit(_detect)
            fut_expense = pool.submit(_expense)
            lines = fut_lines.result() or []
            expense = fut_expense.result() or {}
    else:
        lines = _detect()

    if not textract_enabled():
        return dict(_EMPTY_OCR)

    text = "\n".join(lines)
    demo = _parse_demographics_from_lines(lines)
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
    # Ignore truncated vendor tokens like "DrA"
    if provider and len(re.sub(r"[^A-Za-z]", "", provider)) < 4:
        provider = ""
    patient = (
        demo.get("patient_name")
        or expense.get("NAME")
        or expense.get("CUSTOMER_NAME")
        or expense.get("RECEIVER_NAME")
        or ""
    ).strip()
    if patient and provider and patient.lower() == provider.lower():
        patient = ""

    payment = _extract_payment_fields(text)

    return {
        "gst_number": gst,
        "drug_license_number": "; ".join(licenses) if licenses else "",
        "invoice_number": invoice_number,
        "invoice_date": (expense.get("INVOICE_RECEIPT_DATE") or "").strip(),
        "total_amount": total,
        "provider_name": provider,
        "patient_name": patient,
        "provider_address": (expense.get("VENDOR_ADDRESS") or "").strip(),
        "patient_age": demo.get("patient_age") or "",
        "patient_gender": demo.get("patient_gender") or "",
        "doctor_name": demo.get("doctor_name") or "",
        "clinic_hospital_name": demo.get("clinic_hospital_name") or "",
        "consultation_date": demo.get("consultation_date") or "",
        "diagnosis": demo.get("diagnosis") or "",
        "visual_acuity_details": demo.get("visual_acuity_details") or "",
        "provider_contact": demo.get("provider_contact") or "",
        "payment_mode": payment.get("payment_mode") or "",
        "payment_amount": payment.get("payment_amount") or "",
        "transaction_date": payment.get("transaction_date") or "",
        "transaction_id": payment.get("transaction_id") or "",
        "reference_number": payment.get("reference_number") or "",
        "utr": payment.get("utr") or "",
        "payer_name": payment.get("payer_name") or "",
        "payee_name": payment.get("payee_name") or "",
        "upi_id": payment.get("upi_id") or "",
        "bank_name": payment.get("bank_name") or "",
        "payment_status": payment.get("payment_status") or "",
        "payment_time": payment.get("payment_time") or "",
        "account_number_masked": payment.get("account_number_masked") or "",
        "ifsc": payment.get("ifsc") or "",
        "remarks": payment.get("remarks") or "",
        "raw_text": text,
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
    """Fill empty OpenAI invoice / Rx / report / payment gaps with Textract OCR values."""
    if not ocr:
        return

    def _blank(val: Any) -> bool:
        s = str(val or "").strip()
        return not s or s.lower() == "present"

    inv_raw = data.get("invoice_parameters")
    inv: Dict[str, Any] = dict(inv_raw) if isinstance(inv_raw, dict) else {}
    rx_raw = data.get("prescription_parameters")
    rx: Dict[str, Any] = dict(rx_raw) if isinstance(rx_raw, dict) else {}
    rep_raw = data.get("report_parameters")
    rep: Dict[str, Any] = dict(rep_raw) if isinstance(rep_raw, dict) else {}
    pay_raw = data.get("payment_receipt_parameters")
    pay: Dict[str, Any] = dict(pay_raw) if isinstance(pay_raw, dict) else {}

    invoice_fill_keys = (
        "gst_number",
        "drug_license_number",
        "invoice_number",
        "invoice_date",
        "total_amount",
        "provider_name",
        "patient_name",
        "provider_address",
        "patient_age",
        "patient_gender",
        "doctor_name",
        "provider_contact",
    )
    rx_fill_keys = (
        "patient_name",
        "patient_age",
        "patient_gender",
        "doctor_name",
        "clinic_hospital_name",
        "consultation_date",
        "diagnosis",
        "visual_acuity_details",
    )
    report_fill_keys = (
        "patient_name",
        "patient_age",
        "patient_gender",
    )
    payment_fill_keys = (
        "payment_mode",
        "payment_amount",
        "transaction_date",
        "transaction_id",
        "reference_number",
        "utr",
        "payer_name",
        "payee_name",
        "upi_id",
        "bank_name",
        "payment_status",
        "payment_time",
        "account_number_masked",
        "ifsc",
        "remarks",
    )

    for key in invoice_fill_keys:
        ocr_val = (ocr.get(key) or "").strip()
        if ocr_val and _blank(inv.get(key)):
            inv[key] = ocr_val

    for key in rx_fill_keys:
        ocr_val = (ocr.get(key) or "").strip()
        if ocr_val and _blank(rx.get(key)):
            rx[key] = ocr_val

    for key in report_fill_keys:
        ocr_val = (ocr.get(key) or "").strip()
        if ocr_val and _blank(rep.get(key)):
            rep[key] = ocr_val

    for key in payment_fill_keys:
        ocr_val = (ocr.get(key) or "").strip()
        if ocr_val and _blank(pay.get(key)):
            pay[key] = ocr_val

    # Prefer valid GSTIN from OCR even when OpenAI returned garbage.
    ocr_gst = (ocr.get("gst_number") or "").strip()
    if ocr_gst and _extract_gstin(ocr_gst):
        existing = _extract_gstin(str(inv.get("gst_number") or ""))
        if not existing:
            inv["gst_number"] = _extract_gstin(ocr_gst) or ocr_gst

    ocr_dl = (ocr.get("drug_license_number") or "").strip()
    if ocr_dl and _blank(inv.get("drug_license_number")):
        inv["drug_license_number"] = ocr_dl

    # Map clinic → provider when invoice provider empty (Rx letterhead / OPD summary).
    if _blank(inv.get("provider_name")) and not _blank(rx.get("clinic_hospital_name")):
        inv["provider_name"] = rx.get("clinic_hospital_name")

    if inv:
        data["invoice_parameters"] = inv
    if rx:
        data["prescription_parameters"] = rx
    if rep:
        data["report_parameters"] = rep
    if pay:
        data["payment_receipt_parameters"] = pay

    params_raw = data.get("parameters")
    params: Dict[str, Any] = dict(params_raw) if isinstance(params_raw, dict) else {}
    for key, raw in ocr.items():
        ocr_val = str(raw or "").strip()
        if ocr_val and _blank(params.get(key)):
            params[key] = ocr_val
    if params:
        data["parameters"] = params

    category = str(data.get("document_category", "other"))
    # Strong payment proof only — never flip on a lone email-like "@" token.
    payment_signal = bool(
        (ocr.get("utr") or "").strip()
        or (
            (ocr.get("payment_amount") or "").strip()
            and (ocr.get("payment_status") or "").strip()
            and (ocr.get("payment_mode") or "").strip()
        )
        or (
            _is_plausible_upi_id((ocr.get("upi_id") or "").strip())
            and (ocr.get("payment_amount") or "").strip()
            and (ocr.get("payment_status") or (ocr.get("transaction_id") or "")).strip()
        )
    )
    medical_ocr = bool(
        (ocr.get("doctor_name") or "").strip()
        or (ocr.get("visual_acuity_details") or "").strip()
        or (ocr.get("diagnosis") or "").strip()
        or (
            (ocr.get("consultation_date") or "").strip()
            and (ocr.get("clinic_hospital_name") or "").strip()
        )
    )

    # Payment proof cues override false "other" / accidental invoice labels.
    if (
        payment_signal
        and not medical_ocr
        and category in ("other", "", "invoice")
    ):
        # Don't override a real medical bill that also mentions UPI payment mode.
        looks_bill = bool(
            (ocr.get("gst_number") or "").strip()
            or (
                (ocr.get("invoice_number") or "").strip()
                and (ocr.get("total_amount") or "").strip()
            )
        )
        if category != "invoice" or not looks_bill:
            data["is_medical_document"] = False
            data["document_category"] = "payment_receipt"
            return

    # If OpenAI missed category but expense OCR looks like a bill, nudge recovery.
    if not data.get("is_medical_document", True) and (
        ocr_gst or (ocr.get("invoice_number") and ocr.get("total_amount"))
    ):
        if category != "payment_receipt":
            data["is_medical_document"] = True
            if category in ("other", ""):
                data["document_category"] = "invoice"
