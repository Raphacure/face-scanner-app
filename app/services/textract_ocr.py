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
    r"(?:"
    r"invoice\s*(?:no|number|#)|inv\.?\s*no\.?|bill\s*no\.?|receipt\s*no\.?|"
    r"memo\s*no\.?|cash\s*memo\s*no\.?|serial\s*no\.?|"
    r"बिल\s*(?:नं\.?|नंबर)?|रसीद\s*(?:नं\.?|नंबर)?|पावती\s*(?:नं\.?|नंबर)?|"
    r"क्रमांक|अनुक्रमांक"
    r")"
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
# Pharmacy cash memo: "Prescribed for:...... Paksh Chaudhary" / "By Dr....... Kailash"
_PRESCRIBED_FOR_RE = re.compile(
    r"Prescribed\s*for\s*[:.\-\s]*([A-Za-z][A-Za-z.'\s]{1,60}?)"
    r"(?=\s*(?:By\s*Dr|Doctor|Date|No\.?\b|Particular|$|\n))",
    re.IGNORECASE,
)
# OPD Rx pads commonly print "Name :" (not "Patient Name") next to Age/Sex / Date.
_PATIENT_NAME_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:Patient(?:['’]?s)?\s+|Name\s+of\s+(?:the\s+)?Patient\s+|Pt\.?\s+)?"
    r"Name\s*[:.\-]\s*"
    r"([A-Za-z][A-Za-z.'\s]{1,60}?)"
    r"(?=\s*(?:Age\s*/?\s*Sex|Age|Sex|Date|OPD|Regd?\.?\s*No|Mobile|Phone|"
    r"M\.?R\.?\s*No|CR\s*No|Token|Wt\.?|Weight|Gender|Address|Category|$|\n))",
    re.IGNORECASE,
)
_PATIENT_NAME_STOP_RE = re.compile(
    r"\s+(?:Age\s*/?\s*Sex|Age|Sex|Date|OPD|Regd?\.?\s*No|Mobile|Phone|"
    r"M\.?R\.?\s*No|CR\s*No|Token|Wt\.?|Weight|Gender|Address|Category)\b.*$",
    re.IGNORECASE,
)
# Textract often emits handwriting as its own line ABOVE printed Name:/Age/Sex:/Date:.
_DEMOGRAPHICS_LABEL_LINE_RE = re.compile(
    r"^(?:Patient(?:['’]?s)?\s+)?(?:Name|Age\s*/?\s*Sex|Age|Sex|Date)\s*[:.\-]?$",
    re.IGNORECASE,
)
_DEMOGRAPHICS_NAME_PREFIX_RE = re.compile(
    r"^(?:Patient(?:['’]?s)?\s+)?Name\s*[:.\-]",
    re.IGNORECASE,
)
_PERSON_NAME_LINE_RE = re.compile(
    r"^[A-Za-z][A-Za-z.'\-]{1,30}(?:\s+[A-Za-z][A-Za-z.'\-]{1,30}){0,3}$"
)
_HOSPITALISH_NAME_RE = re.compile(
    r"\b(?:hospital|clinic|pharmacy|medical\s+(?:store|centre|center)|"
    r"nursing\s*home|multispeciality|multi[\s\-]?specialty)\b",
    re.IGNORECASE,
)
_NON_PERSON_NAME_TOKENS = frozenset(
    {
        "age",
        "sex",
        "date",
        "name",
        "patient",
        "hospital",
        "clinic",
        "pharmacy",
        "physician",
        "doctor",
        "dr",
        "mbbs",
        "timings",
        "medical",
        "store",
        "nursing",
        "home",
        "general",
        "opd",
    }
)
_BY_DR_RE = re.compile(
    r"By\s*Dr\.?\s*[:.\-\s]*([A-Za-z][A-Za-z.'\s]{1,60}?)"
    r"(?=\s*(?:Date|No\.?\b|Particular|Prescribed|Age|Sex|$|\n))",
    re.IGNORECASE,
)
_FACILITY_LINE_RE = re.compile(
    r"(?:^|\n)\s*Facility\s*[:\-]?\s*([^\n]{3,80})",
    re.IGNORECASE,
)
# Generic English facility title on letterhead (any hospital/clinic name).
_HOSPITAL_TITLE_RE = re.compile(
    r"(?:^|\n)\s*([A-Za-z0-9][A-Za-z0-9 .,&'\-]{0,70}?"
    r"(?:Hospitals?|Clinic|Multispeciality|Multi[\s\-]?Specialty|Nursing\s*Home|"
    r"Medical\s*Centre|Medical\s*Center|Polyclinic|"
    r"Institute(?:\s+of\s+Medical\s+Sciences)?|Medical\s+Sciences))\b",
    re.IGNORECASE,
)
# Logo OCR often splits brand + HOSPITALS across two lines (e.g. SRI\nHOSPITALS).
_SPLIT_HOSPITAL_LOGO_RE = re.compile(
    r"(?:^|\n)\s*([A-Za-z][A-Za-z0-9.&']{1,24})\s*\n\s*"
    r"(Hospitals?|Clinic|Multispeciality|Multi[\s\-]?Specialty|"
    r"Nursing\s*Home|Medical\s*(?:Centre|Center)|Polyclinic|"
    r"Institute(?:\s+of\s+Medical\s+Sciences)?)\b",
    re.IGNORECASE,
)
# Regional-script hospital words (Tamil/Hindi/etc.) — map to English when logo OCR fails.
_REGIONAL_HOSPITAL_HINT_RE = re.compile(
    r"(மருத்துவமனை|கிளினிக்|अस्पताल|हॉस्पिटल|क्लिनिक|"
    r"హాస్పిటల్|క్లినిక్|ಆಸ್ಪತ್ರೆ|ആശുപത്രി|হাসপাতাল)",
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


def _extract_hospital_title(text: str) -> str:
    """Letterhead hospital/clinic title from one line, split logo lines, or regional script."""
    if not text:
        return ""
    # Prefer English logo split across lines (SRI\nHOSPITALS) — common on circular logos.
    split = _SPLIT_HOSPITAL_LOGO_RE.search(text)
    if split:
        brand = split.group(1).strip(" .,-")
        kind = split.group(2).strip(" .,-")
        # Skip noise tokens that are not brand names.
        if brand and brand.lower() not in {"the", "and", "for", "dr", "born", "save"}:
            return f"{brand} {kind}".strip()
    hm = _HOSPITAL_TITLE_RE.search(text)
    if hm:
        return hm.group(1).strip(" .,-")
    # Standalone HOSPITALS line with a short brand line immediately above.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for idx, line in enumerate(lines):
        if re.fullmatch(r"Hospitals?", line, re.IGNORECASE) and idx > 0:
            prev = lines[idx - 1]
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9.&']{1,24}", prev):
                return f"{prev} {line}"
    # Regional script present but English OCR failed — still signal a hospital letterhead.
    # Callers/OpenAI should prefer a proper English name; Textract may only see Tamil glyphs.
    if _REGIONAL_HOSPITAL_HINT_RE.search(text):
        # Try any nearby Latin brand token before the regional hospital word.
        for m in _REGIONAL_HOSPITAL_HINT_RE.finditer(text):
            window = text[max(0, m.start() - 80) : m.start()]
            latin = re.findall(r"\b([A-Z][A-Za-z0-9.&']{1,20})\b", window)
            if latin:
                brand = latin[-1]
                if brand.lower() not in {"dr", "date", "name", "age", "regn", "reg"}:
                    return f"{brand} Hospital"
        return "Hospital"
    return ""


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
    # Longer labels first so "Patient Name" wins over bare "Patient" / "Name".
    ordered = sorted(labels, key=lambda lab: len(lab), reverse=True)
    label_set = {lab.lower() for lab in ordered}
    for idx, line in enumerate(lines):
        raw = line.strip()
        low = re.sub(r"[:.\-\s]+$", "", raw.lower())
        if low in label_set:
            if idx + 1 < len(lines):
                nxt = lines[idx + 1].strip().lstrip(".:- ")
                if nxt and re.sub(r"[:.\-\s]+$", "", nxt.lower()) not in label_set:
                    return nxt
            continue
        for lab in ordered:
            lab_low = lab.lower()
            # Bare "Name" must be a label token (Name / Name:) — not "Named …".
            if lab_low == "name":
                if not low.startswith("name") or re.match(r"^name[a-z]", low):
                    continue
            elif lab_low == "clinic":
                # Do not match "Clinical Notes/Prescriptions" as a Clinic label.
                if not (
                    low == "clinic"
                    or low.startswith("clinic ")
                    or low.startswith("clinic:")
                    or low.startswith("clinic-")
                    or low.startswith("clinic.")
                ):
                    continue
            elif lab_low == "hospital":
                # Do not treat standalone "HOSPITALS" logo line as Hospital:<value>.
                if not (
                    low == "hospital"
                    or low.startswith("hospital ")
                    or low.startswith("hospital:")
                    or low.startswith("hospital-")
                    or low.startswith("hospital.")
                ):
                    continue
            elif not low.startswith(lab_low):
                continue
            # Same-line value after "Prescribed for:...... Name" / "By Dr....... Name"
            rest = raw[len(lab) :].lstrip(" :.-")
            if rest:
                return rest
            if idx + 1 < len(lines):
                nxt = lines[idx + 1].strip().lstrip(".:- ")
                if nxt and re.sub(r"[:.\-\s]+$", "", nxt.lower()) not in label_set:
                    return nxt
    return ""


def _clean_patient_name_value(raw: str) -> str:
    """Strip trailing Age/Sex/Date tails from a same-line Name capture."""
    name = (raw or "").strip(" .:-")
    if not name:
        return ""
    name = _PATIENT_NAME_STOP_RE.sub("", name).strip(" .:-")
    # Reject non-name leftovers (dates, amounts, single tokens that are labels).
    if not name or re.fullmatch(r"[\d./\-]+", name):
        return ""
    if re.match(r"^(?:age|sex|date|name|patient)\b", name, re.IGNORECASE):
        return ""
    if name.lower() in _NON_PERSON_NAME_TOKENS:
        return ""
    if _HOSPITALISH_NAME_RE.search(name):
        return ""
    tokens = re.findall(r"[A-Za-z]+", name)
    if not tokens or any(tok.lower() in _NON_PERSON_NAME_TOKENS for tok in tokens):
        return ""
    if len(name) < 3:
        return ""
    return name


def _looks_like_person_name(raw: str) -> bool:
    """True for a short alphabetic person-name line (not hospital / label junk)."""
    name = _clean_patient_name_value(raw)
    if not name or not _PERSON_NAME_LINE_RE.match(name):
        return False
    tokens = name.split()
    # Prefer multi-token Indian names; allow long single tokens (e.g. "Susheela").
    if len(tokens) == 1 and len(tokens[0]) < 5:
        return False
    return True


def _orphan_patient_name_near_demographics(lines: List[str]) -> str:
    """Recover handwriting Textract placed above/below printed Name:/Age/Sex:/Date:."""
    demo_indexes: List[int] = []
    for idx, line in enumerate(lines):
        raw = line.strip()
        if not raw:
            continue
        if _DEMOGRAPHICS_LABEL_LINE_RE.match(raw) or _DEMOGRAPHICS_NAME_PREFIX_RE.match(raw):
            demo_indexes.append(idx)
    if not demo_indexes:
        return ""
    # Prefer the Name label when present; otherwise first demographics cue.
    name_idx = next(
        (
            i
            for i in demo_indexes
            if re.match(r"^(?:Patient(?:['’]?s)?\s+)?Name\b", lines[i].strip(), re.I)
        ),
        demo_indexes[0],
    )

    def _scan(indexes: Sequence[int]) -> str:
        for j in indexes:
            if j < 0 or j >= len(lines):
                continue
            cand = lines[j].strip()
            if not cand or _DEMOGRAPHICS_LABEL_LINE_RE.match(cand):
                continue
            if _DEMOGRAPHICS_NAME_PREFIX_RE.match(cand):
                # "Name : X" — ignore OCR junk on the label line itself.
                continue
            if re.fullmatch(r"[\d./\-():\s]+", cand):
                continue
            if _looks_like_person_name(cand):
                return _clean_patient_name_value(cand)
        return ""

    # Handwriting is usually above the printed labels; also try a short window below.
    found = _scan(range(name_idx - 1, max(-1, name_idx - 6), -1))
    if found:
        return found
    return _scan(range(name_idx + 1, min(len(lines), name_idx + 5)))


def _append_patient_name_continuation(lines: List[str], patient: str) -> str:
    """Join a second handwritten name line under Patient Name (e.g. Lakshmi)."""
    base = _clean_patient_name_value(patient)
    if not base:
        return ""
    name_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(
                r"^(?:Patient(?:['’]?s)?\s+)?Name\s*[:.\-]",
                line.strip(),
                re.IGNORECASE,
            )
        ),
        None,
    )
    if name_idx is None:
        return base
    name_line = lines[name_idx].strip()
    # Only continue when the labeled Name line already holds the first part
    # (avoids appending Rx body tokens like "SOB" after orphan-name recovery).
    first_token = base.split()[0].lower()
    if first_token not in name_line.lower():
        return base
    for j in range(name_idx + 1, min(len(lines), name_idx + 5)):
        cand = lines[j].strip()
        if not cand:
            continue
        if _DEMOGRAPHICS_LABEL_LINE_RE.match(cand) or re.match(
            r"^(?:Time|Wt\.?|Weight|Rx|Temp|Date|Address|Category|Mobile|Department|"
            r"Doctor|Room|OPD|CR\s*No|Card|ALERTS?|DIAGNOSIS|Investigations?)\b",
            cand,
            re.IGNORECASE,
        ):
            continue
        if re.fullmatch(r"[\d./\-():\s]+", cand):
            continue
        # Stop at labeled rows like "Address : ..." / "Category Paying".
        if re.match(r"^[A-Za-z][A-Za-z /]{1,20}\s*[:\-]", cand):
            continue
        cont = _clean_patient_name_value(cand)
        if not cont or not _PERSON_NAME_LINE_RE.match(cont):
            continue
        # Only append short leftover tokens (surname / last part), not a new full name.
        if len(cont.split()) != 1:
            continue
        if cont.lower() in base.lower():
            continue
        # Reject common form words OCR'd as a "name" continuation.
        if cont.lower() in _NON_PERSON_NAME_TOKENS | {
            "address",
            "category",
            "paying",
            "mobile",
            "department",
            "doctor",
            "room",
            "alerts",
            "diagnosis",
        }:
            continue
        return f"{base} {cont}".strip()
    return base


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

    patient = _value_after_label(
        lines,
        [
            "Patient Name",
            "Patient's Name",
            "Name of Patient",
            "Name of the Patient",
            "Beneficiary Name",
            "Pt. Name",
            "Pt Name",
            "Patient",
            "Prescribed for",
            "Prescribed For",
            "Name",
        ],
    )
    # Same-line "Name : Age / Sex : ..." means the handwritten value was not OCR'd
    # on that line — do not treat the next printed label as the patient name.
    if patient and re.match(r"^(?:age|sex|date)\b", patient.strip(), re.IGNORECASE):
        patient = ""
    if not patient:
        nm = _PATIENT_NAME_LINE_RE.search(text)
        patient = nm.group(1).strip(" .") if nm else ""
    if not patient:
        pm = _PRESCRIBED_FOR_RE.search(text)
        patient = pm.group(1).strip(" .") if pm else ""
    patient = _clean_patient_name_value(patient)
    if not patient:
        patient = _orphan_patient_name_near_demographics(lines)
    if patient:
        patient = _append_patient_name_continuation(lines, patient)
        out["patient_name"] = patient

    doctor = _value_after_label(
        lines, ["Doctor", "Consultant", "Unit Consultants", "By Dr", "By Dr."]
    )
    if not doctor:
        dm = _BY_DR_RE.search(text) or _DOCTOR_LINE_RE.search(text)
        doctor = dm.group(1).strip(" .") if dm else ""
    if doctor and len(doctor) > 3:
        out["doctor_name"] = doctor

    facility = _value_after_label(lines, ["Facility", "Hospital", "Clinic"])
    if facility and (
        len(facility) < 4
        or re.search(r"\bnotes?/prescriptions?\b", facility, re.I)
        or re.match(r"^(?:clinical|investigations?|diagnosis|alerts?)\b", facility, re.I)
        or re.fullmatch(r"[A-Za-z]", facility)
    ):
        facility = ""
    if not facility:
        fm = _FACILITY_LINE_RE.search(text)
        facility = fm.group(1).strip() if fm else ""
    if not facility:
        facility = _extract_hospital_title(text)
    if facility and len(facility) > 3:
        # Skip doctor qualification lines mistaken as hospital titles.
        if not re.search(r"\b(?:M\.?B\.?B\.?S|F\.?C\.?C\.?M|F\.?D\.?M)\b", facility, re.I):
            if not re.search(r"\bnotes?/prescriptions?\b", facility, re.I):
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
    patient = _clean_patient_name_value(demo.get("patient_name") or "")
    if not patient:
        for key in ("NAME", "CUSTOMER_NAME", "RECEIVER_NAME"):
            candidate = _clean_patient_name_value(expense.get(key) or "")
            if candidate:
                patient = candidate
                break
    clinic = (demo.get("clinic_hospital_name") or "").strip()
    if patient and provider and patient.lower() == provider.lower():
        patient = ""
    if patient and clinic and patient.lower() == clinic.lower():
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


def _usable_patient_name(raw: Any) -> str:
    """Return a cleaned person name, or "" if empty / hospital / placeholder."""
    name = _clean_patient_name_value(str(raw or ""))
    if not name:
        return ""
    low = name.lower().strip()
    if low in {
        "n/a",
        "na",
        "nil",
        "none",
        "null",
        "-",
        "--",
        "unknown",
        "not available",
        "not applicable",
        "not visible",
        "missing",
        "present",
    }:
        return ""
    if _HOSPITALISH_NAME_RE.search(name):
        return ""
    return name


def sync_patient_name_across_buckets(data: Dict[str, Any]) -> None:
    """Copy shared identity fields into every claim bucket (stops false missing escalations)."""
    if not isinstance(data, dict):
        return
    buckets: List[Dict[str, Any]] = []
    for key in (
        "parameters",
        "prescription_parameters",
        "invoice_parameters",
        "report_parameters",
    ):
        raw = data.get(key)
        if isinstance(raw, dict):
            buckets.append(raw)
        else:
            bucket: Dict[str, Any] = {}
            data[key] = bucket
            buckets.append(bucket)

    _PLACEHOLDER = {
        "n/a",
        "na",
        "nil",
        "none",
        "null",
        "-",
        "--",
        "unknown",
        "not available",
        "not applicable",
        "not visible",
        "missing",
    }

    def _is_empty(val: Any, *, treat_present_as_filled: bool = False) -> bool:
        s = str(val or "").strip()
        if not s:
            return True
        low = s.lower()
        if treat_present_as_filled and low == "present":
            return False
        return low in _PLACEHOLDER or low == "present"

    def _best(key: str, *, patient: bool = False) -> str:
        best = ""
        for bucket in buckets:
            raw_val = bucket.get(key)
            if patient:
                cand = _usable_patient_name(raw_val)
            else:
                cand = str(raw_val or "").strip()
                if _is_empty(cand, treat_present_as_filled=(key in {
                    "doctor_signature",
                    "doctor_stamp",
                    "authorized_stamp",
                    "authorized_signature",
                })):
                    cand = ""
            if cand and (not best or len(cand) > len(best)):
                best = cand
        return best

    def _fill(key: str, value: str, *, patient: bool = False) -> None:
        if not value:
            return
        for bucket in buckets:
            cur = bucket.get(key)
            if patient:
                if _usable_patient_name(cur):
                    continue
                bucket[key] = value
                continue
            if _is_empty(
                cur,
                treat_present_as_filled=(
                    key
                    in {
                        "doctor_signature",
                        "doctor_stamp",
                        "authorized_stamp",
                        "authorized_signature",
                    }
                ),
            ):
                bucket[key] = value

    _fill("patient_name", _best("patient_name", patient=True), patient=True)
    for key in (
        "patient_age",
        "patient_gender",
        "doctor_name",
        "doctor_registration_number",
        "consultation_date",
        "clinic_hospital_address",
        "provider_address",
        "laboratory_address",
    ):
        _fill(key, _best(key))

    facility = (
        _best("clinic_hospital_name")
        or _best("provider_name")
        or _best("laboratory_name")
    )
    if facility:
        for key in ("clinic_hospital_name", "provider_name", "laboratory_name"):
            _fill(key, facility)

    for key in (
        "doctor_signature",
        "doctor_stamp",
        "authorized_stamp",
        "authorized_signature",
    ):
        _fill(key, _best(key))


def fill_demographics_from_text(data: Dict[str, Any], text: str) -> None:
    """Fill empty patient/clinic demographics from OCR or PDF text into all buckets."""
    if not isinstance(data, dict) or not (text or "").strip():
        return
    demo = _parse_demographics_from_lines(text.splitlines())
    patient = _usable_patient_name(demo.get("patient_name"))
    clinic = str(demo.get("clinic_hospital_name") or "").strip()
    doctor = str(demo.get("doctor_name") or "").strip()
    age = str(demo.get("patient_age") or "").strip()
    gender = str(demo.get("patient_gender") or "").strip()
    consult = str(demo.get("consultation_date") or "").strip()

    def _blank(val: Any) -> bool:
        s = str(val or "").strip()
        return not s or s.lower() in {
            "n/a",
            "na",
            "nil",
            "none",
            "null",
            "-",
            "--",
            "unknown",
            "not available",
            "not applicable",
            "not visible",
            "missing",
            "present",
        }

    for key in (
        "parameters",
        "prescription_parameters",
        "invoice_parameters",
        "report_parameters",
    ):
        raw = data.get(key)
        bucket: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        if patient and not _usable_patient_name(bucket.get("patient_name")):
            bucket["patient_name"] = patient
        if clinic and _blank(bucket.get("clinic_hospital_name")):
            if key in ("parameters", "prescription_parameters"):
                bucket["clinic_hospital_name"] = clinic
        if clinic and _blank(bucket.get("provider_name")) and key in (
            "parameters",
            "invoice_parameters",
        ):
            bucket["provider_name"] = clinic
        if clinic and _blank(bucket.get("laboratory_name")) and key in (
            "parameters",
            "report_parameters",
        ):
            bucket["laboratory_name"] = clinic
        if doctor and _blank(bucket.get("doctor_name")):
            bucket["doctor_name"] = doctor
        if age and _blank(bucket.get("patient_age")):
            bucket["patient_age"] = age
        if gender and _blank(bucket.get("patient_gender")):
            bucket["patient_gender"] = gender
        if consult and _blank(bucket.get("consultation_date")):
            bucket["consultation_date"] = consult
        data[key] = bucket

    sync_patient_name_across_buckets(data)


def merge_textract_into_openai_data(
    data: Dict[str, Any],
    ocr: Dict[str, str],
) -> None:
    """Fill empty OpenAI invoice / Rx / report / payment gaps with Textract OCR values."""
    if not ocr:
        return

    _PLACEHOLDER_LOW = frozenset(
        {
            "n/a",
            "na",
            "nil",
            "none",
            "null",
            "-",
            "--",
            "unknown",
            "not available",
            "not applicable",
            "not visible",
            "missing",
            "present",
        }
    )

    def _blank(val: Any) -> bool:
        s = str(val or "").strip()
        return not s or s.lower() in _PLACEHOLDER_LOW

    def _should_fill_patient(existing: Any, ocr_val: str) -> bool:
        if not _clean_patient_name_value(ocr_val):
            return False
        cur = str(existing or "").strip()
        return _blank(cur) or bool(_HOSPITALISH_NAME_RE.search(cur))

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
        if not ocr_val:
            continue
        if key == "patient_name":
            if _should_fill_patient(inv.get(key), ocr_val):
                inv[key] = ocr_val
            continue
        if _blank(inv.get(key)):
            inv[key] = ocr_val

    for key in rx_fill_keys:
        ocr_val = (ocr.get(key) or "").strip()
        if not ocr_val:
            continue
        if key == "patient_name":
            if _should_fill_patient(rx.get(key), ocr_val):
                rx[key] = ocr_val
            continue
        if key == "clinic_hospital_name":
            existing_clinic = str(rx.get(key) or "").strip()
            junk_clinic = bool(
                re.search(r"\bnotes?/prescriptions?\b", existing_clinic, re.I)
            )
            if (
                (_blank(existing_clinic) or junk_clinic)
                and ocr_val
                and not re.search(r"\bnotes?/prescriptions?\b", ocr_val, re.I)
            ):
                rx[key] = ocr_val
            continue
        if _blank(rx.get(key)):
            rx[key] = ocr_val

    for key in report_fill_keys:
        ocr_val = (ocr.get(key) or "").strip()
        if not ocr_val:
            continue
        if key == "patient_name":
            if _should_fill_patient(rep.get(key), ocr_val):
                rep[key] = ocr_val
            continue
        if _blank(rep.get(key)):
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
        if not ocr_val:
            continue
        if key == "patient_name":
            if _should_fill_patient(params.get(key), ocr_val):
                params[key] = ocr_val
            continue
        if ocr_val and _blank(params.get(key)):
            params[key] = ocr_val
    if params:
        data["parameters"] = params

    sync_patient_name_across_buckets(data)

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
