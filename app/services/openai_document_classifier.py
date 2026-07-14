"""
Classify medical document images and extract structured parameters
for prescription (Rx), invoice, and diagnostic report.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor
from openai import APIStatusError, RateLimitError

from app.aws.textract_client import textract_enabled
from app.core.openai_client import get_openai_client
from app.services.document_image_fetch import (
    DocumentPages,
    build_gst_header_blocks_from_document,
    build_header_crop_from_document,
    build_regulatory_header_blocks,
    build_stamp_blocks_from_document,
    build_vision_blocks_from_document,
    load_document,
)
from app.services.textract_ocr import (
    extract_textract_fields,
    merge_textract_into_openai_data,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_RETRIES = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "6")))
OPENAI_INTER_CALL_DELAY_MS = max(0, int(os.getenv("OPENAI_INTER_CALL_DELAY_MS", "0")))

_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}

_MEDICINE_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "medicine": {"type": "string"},
        "dosage": {"type": "string"},
    },
    "required": ["medicine", "dosage"],
    "additionalProperties": False,
}

PRESCRIPTION_PARAM_KEYS: Tuple[str, ...] = (
    "patient_name",
    "patient_age",
    "patient_gender",
    "consultation_date",
    "clinic_hospital_name",
    "clinic_hospital_address",
    "doctor_name",
    "doctor_qualification",
    "doctor_registration_number",
    "doctor_signature",
    "doctor_stamp",
    "diagnosis",
    "presenting_complaints",
    "line_of_treatment",
    "prescribed_medicines",
    "advised_tests",
    "treatment_plan",
    "followup_date",
    "follow_up_advice",
    "affected_tooth_number",
    "treatment_advised",
    "procedure_recommendation",
    "visual_acuity_details",
    "eye_power_prescription",
    "treatment_advice",
    "glasses_contact_lens_prescription",
)

INVOICE_PARAM_KEYS: Tuple[str, ...] = (
    "patient_name",
    "patient_age",
    "patient_gender",
    "invoice_number",
    "invoice_date",
    "sample_collection_date",
    "provider_name",
    "provider_address",
    "provider_contact",
    "doctor_name",
    "service_details",
    "consultation_charges",
    "registration_charges",
    "medicine_details",
    "test_details",
    "item_details",
    "total_amount",
    "payment_mode",
    "transaction_reference",
    "gst_number",
    "drug_license_number",
    "authorized_stamp",
    "authorized_signature",
)

REPORT_PARAM_KEYS: Tuple[str, ...] = (
    "patient_name",
    "patient_age",
    "patient_gender",
    "laboratory_name",
    "laboratory_address",
    "test_names",
    "sample_collection_date",
    "report_date",
    "test_results",
    "reference_ranges",
    "pathologist_name",
    "pathologist_registration_number",
    "pathologist_signature",
    "authorized_stamp",
)

# Shared required fields for most prescription subtypes.
PRESCRIPTION_REQUIRED: Tuple[str, ...] = (
    "patient_name",
    "consultation_date",
    "clinic_hospital_name",
    "doctor_name",
    "doctor_registration_number",
    "doctor_signature",
    "doctor_stamp",
)

PRESCRIPTION_SUBTYPE_REQUIRED: Dict[str, Tuple[str, ...]] = {
    "opd": PRESCRIPTION_REQUIRED,
    "pharmacy": (
        "consultation_date",
        "clinic_hospital_name",
        "doctor_name",
        "doctor_registration_number",
        "doctor_signature",
        "doctor_stamp",
    ),
    "diagnostic": PRESCRIPTION_REQUIRED,
    "dental": PRESCRIPTION_REQUIRED,
    "eye_care": PRESCRIPTION_REQUIRED,
}

INVOICE_REQUIRED: Tuple[str, ...] = (
    "patient_name",
    "invoice_number",
    "invoice_date",
    "provider_name",
    "total_amount",
)

REPORT_REQUIRED: Tuple[str, ...] = (
    "patient_name",
    "laboratory_name",
    "test_names",
    "sample_collection_date",
    "report_date",
    "test_results",
    "pathologist_registration_number",
)


def _unique_param_keys(*groups: Sequence[str]) -> Tuple[str, ...]:
    """Stable union of param keys across prescription / invoice / report."""
    seen: set[str] = set()
    ordered: List[str] = []
    for group in groups:
        for key in group:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return tuple(ordered)


# Full fixed schema for main-service presence checks (all claim document types).
ALL_CLAIM_PARAM_KEYS: Tuple[str, ...] = _unique_param_keys(
    PRESCRIPTION_PARAM_KEYS,
    INVOICE_PARAM_KEYS,
    REPORT_PARAM_KEYS,
)

_INVOICE_ARRAY_KEYS = frozenset(
    {"service_details", "medicine_details", "test_details", "item_details"}
)
_INVOICE_DETAIL_LIST_KEYS = frozenset(
    {"medicine_details", "test_details", "item_details"}
)
_VALID_INVOICE_SUBTYPES = frozenset(
    {
        "pharmacy",
        "diagnostic",
        "opd_consultation",
        "dental",
        "eye_care",
        "uncertain",
        "not_applicable",
    }
)
_VALID_PRESCRIPTION_SUBTYPES = frozenset(
    {
        "opd",
        "pharmacy",
        "diagnostic",
        "dental",
        "eye_care",
        "uncertain",
        "not_applicable",
    }
)
_PRESCRIPTION_ARRAY_KEYS = frozenset({"advised_tests"})
_REPORT_ARRAY_KEYS = frozenset({"test_names", "test_results", "reference_ranges"})
_ALL_CLAIM_ARRAY_KEYS = (
    _PRESCRIPTION_ARRAY_KEYS | _INVOICE_ARRAY_KEYS | _REPORT_ARRAY_KEYS
)


def _empty_all_claim_parameters() -> Dict[str, Any]:
    """Every extractable field, empty — fixed shape for main-service validation."""
    return _normalize_params({}, ALL_CLAIM_PARAM_KEYS, _ALL_CLAIM_ARRAY_KEYS)


def _to_all_claim_parameters(partial: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay category extraction onto the full claim field schema."""
    return _normalize_params(partial, ALL_CLAIM_PARAM_KEYS, _ALL_CLAIM_ARRAY_KEYS)


_DOCTOR_REG_IN_TEXT_RE = re.compile(
    r"(?:RMC|MCI|MMC|HPMC|HIMC|DMC|CN\.?\s*NO\.?|C\.?\s*N\.?\s*NO\.?|"
    r"council\s*(?:reg(?:istration)?|no\.?)|reg\.?\s*no\.?|regd\.?\s*no\.?|"
    r"registration\s*no\.?)\s*[:\s#-]*([A-Za-z0-9][A-Za-z0-9/\-]{1,20})",
    re.IGNORECASE,
)
_DOCTOR_REG_DIGITS_RE = re.compile(r"\b(\d{4,8}(?:/\d{2,6})?)\b")
# Explicit CN No. capture (Himachal / state OPD stamps)
_CN_NO_RE = re.compile(
    r"\bC\.?\s*N\.?\s*(?:NO\.?|NUMBER|#)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9/\-]{1,20})",
    re.IGNORECASE,
)
# Himachal / eHospital patient Case Registration (CR No) — never doctor reg
_CR_NO_LIKE_RE = re.compile(r"^\d{12,20}$")

# Indian pharmacy bill header identifiers (GSTIN + state drug license).
_GSTIN_RE = re.compile(
    r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b",
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

_PLACEHOLDER_VALUES = frozenset(
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
    }
)

# Cues for fully printed POS / tax invoices — NOT provider names like "pharmacy".
_PRINTED_POS_INVOICE_SIGNALS = frozenset(
    {
        "tax invoice",
        "retail invoice",
        "bill of supply",
        "serial invoice",
        "invoice no",
        "inv no",
        "inv.no",
    }
)

_PRINTED_MEDICINE_LINE_RE = re.compile(
    r"\b(?:batch|exp(?:iry)?|mrp|hsn|gst\s*%|rack)\b",
    re.IGNORECASE,
)
_HSN_CODE_LINE_RE = re.compile(r"^\d{4,8}\s+[A-Z]", re.IGNORECASE)
_TYPED_PRODUCT_RE = re.compile(
    r"\b(?:TAB|CAP|SYP|SYR|ML|MG|INJ|DT|DROPS?|OINT|CREAM|GEL|LOTION)\b",
    re.IGNORECASE,
)
_DIGITAL_PAYMENT_HINTS = frozenset(
    {
        "digital",
        "upi",
        "card",
        "online",
        "net banking",
        "neft",
        "credit",
        "debit",
        "paytm",
        "gpay",
        "phonepe",
        "cashless",
    }
)

# Pattern heuristics for printed form rows / section titles (not fixed test-name lists).
_INVOICE_PRINTED_ROW_RE = re.compile(
    r"^(?:\d+[\).:\s-]+)?"
    r"(?:(?:blood|urine|stool|sputum|serum|faeces|feces)\s+examination"
    r"|examination\s+of\s+(?:blood|urine|stool|sputum|serum))"
    r"|^(?:particulars|others?|total|sub\s*-?\s*total)\b",
    re.IGNORECASE,
)


_PHARMACY_BILL_HINTS = frozenset(
    {
        "pharmacy",
        "chemist",
        "medical store",
        "bill of supply",
        "medicines",
        "drug license",
        "dl no",
        "gst",
        "pharmacist",
    }
)

_MEDICAL_CLAIM_HINTS = _PHARMACY_BILL_HINTS | frozenset(
    {
        "consultation",
        "clinic",
        "received with thanks",
        "physician",
        "doctor",
        "receipt",
        "hospital",
        "opd",
        "mbbs",
        "dental",
        "dentist",
        "optical",
        "optician",
        "eye care",
        "lens",
        "rotated",
        "upside",
        "inverted",
    }
)


def _param_object_schema(
    keys: Sequence[str],
    array_keys: frozenset[str],
) -> Dict[str, Any]:
    props = {key: _STRING_ARRAY if key in array_keys else {"type": "string"} for key in keys}
    return {
        "type": "object",
        "properties": props,
        "required": list(keys),
        "additionalProperties": False,
    }


def _prescription_params_schema() -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "prescribed_medicines": {"type": "array", "items": _MEDICINE_ITEM_SCHEMA},
    }
    for key in PRESCRIPTION_PARAM_KEYS:
        if key != "prescribed_medicines":
            props[key] = (
                _STRING_ARRAY if key in _PRESCRIPTION_ARRAY_KEYS else {"type": "string"}
            )
    return {
        "type": "object",
        "properties": props,
        "required": list(PRESCRIPTION_PARAM_KEYS),
        "additionalProperties": False,
    }


PRESCRIPTION_PARAMS_SCHEMA = _prescription_params_schema()
INVOICE_PARAMS_SCHEMA = _param_object_schema(INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS)
REPORT_PARAMS_SCHEMA = _param_object_schema(REPORT_PARAM_KEYS, _REPORT_ARRAY_KEYS)

SYSTEM_PROMPT = """Indian medical claims OCR. Fill JSON; use "" / [] if absent. Signatures/stamps: "present" if visible but illegible.

CONTENT TYPE (filled data only; ignore blank form lines). percents must sum to 100.
- handwritten: mostly hand-filled (cash memo) → hw%~100
- computer_generated: typed/printed → cg%~90-100; ignore footer signature alone
- Printed pharmacy/tax invoice/POS = computer_generated. Printed lab report = computer_generated.

CATEGORY
- prescription | invoice | report | other
- other = only screenshots/selfies/blank/unreadable — NEVER pharmacy, clinic, or hospital bills
- is_medical_document=true for Rx, bills, lab reports, OPD receipts
- Rx with no ₹ total → prescription (never invoice). Upside-down photos still medical.

invoice_subtype: pharmacy | diagnostic | opd_consultation | dental | eye_care | uncertain | not_applicable
- pharmacy/chemist Bill of Supply/tax invoice/cash memo → pharmacy
- diagnostic centre bill → diagnostic; clinic "Received with thanks"/OPD fee → opd_consultation
- dental clinic → dental; optical/lens/frame → eye_care

INVOICE: patient_*, invoice_number/date, provider_*, doctor_name, total_amount (digits only, primary total),
payment_mode, transaction_reference, authorized_stamp/signature.
MUST extract gst_number (15-char GSTIN only, no label) from top header/top-right if printed.
GSTIN may appear as **GSTIN:** **02AAHCHxxxxxZx** — strip * and return the code. Never FSSAI/CST/RST.
Pharmacy: drug_license_number (all DL lines, join "; "); medicine_details[] "name | Qty | Rate | Batch | Exp".
OPD: consultation_charges, registration_charges, service_details[].
Diagnostic: sample_collection_date, test_details[] "Test — Rs amt".
Dental/eye: item_details[] procedures or frame/lens lines.

prescription_subtype: opd | pharmacy | diagnostic | dental | eye_care | uncertain | not_applicable
Common: patient_*, consultation_date, clinic_hospital_*, doctor_name/qualification/registration_number,
doctor_signature/stamp, diagnosis, presenting_complaints, line_of_treatment, followup_*.
- opd/pharmacy: prescribed_medicines[{medicine,dosage}]; advised_tests if labs
- diagnostic: advised_tests[]; dental: tooth/treatment/procedure; eye: VA/power/glasses
doctor_registration_number: read from rubber stamp / letterhead — labels include CN No, CN No.,
Council No, Reg No, Regd No, RMC/MCI/MMC/HPMC. Copy the number only (e.g. CN No: 12345 → "12345").
Do NOT use CR No / Patient Registration / Token No / Mobile as doctor_registration_number.
doctor_stamp / doctor_signature: "present" if visible but illegible; still try to read CN/Reg from stamp text.

REPORT: specific test_names (not section titles), test_results, dates, pathologist_*, laboratory_*.
"""

PHARMACY_REGULATORY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "gst_number": {"type": "string"},
        "drug_license_number": {"type": "string"},
    },
    "required": ["gst_number", "drug_license_number"],
    "additionalProperties": False,
}

PHARMACY_GST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"gst_number": {"type": "string"}},
    "required": ["gst_number"],
    "additionalProperties": False,
}

INVOICE_GST_SCHEMA = PHARMACY_GST_SCHEMA

INVOICE_GST_PROMPT = """Extract gst_number ONLY from the TOP header / letterhead / top-right margin.
Return the 15-char GSTIN code only (no label). Format: 2 digits + 5 letters + 4 digits + letter + alnum + Z + alnum.
13th character is ALWAYS Z (not digit 2).
GSTIN may be printed as **GSTIN:** **02AAHCH0054K1ZV** — strip asterisks, return 02AAHCH0054K1ZV.
Do NOT return FSSAI, CST, RST, or drug licence numbers. Use "" only if GSTIN is not printed."""

PHARMACY_REGULATORY_PROMPT = """Pharmacy / chemist cash-memo header.
gst_number: 15-char GSTIN only (13th=Z). Often top-right, sometimes between **asterisks** next to GSTIN:.
Never FSSAI / CST / RST. Strip * from the code.
drug_license_number: every DL/Form 20/21 line; join with "; ".
"" if a field is not printed."""

DOCTOR_REG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "doctor_registration_number": {"type": "string"},
        "doctor_stamp": {"type": "string"},
    },
    "required": ["doctor_registration_number", "doctor_stamp"],
    "additionalProperties": False,
}

DOCTOR_REG_PROMPT = """Extract doctor_registration_number from the doctor rubber stamp / signature area
(usually bottom or bottom-right of an OPD / prescription card).
Labels include: CN No, CN No., CN-, C.N. No, Council No, Reg No, Regd No, RMC, MCI, MMC, HPMC.
Return the registration code only — e.g. printed/written "CN-2158" → "2158"; "CN No: 12345" → "12345".
Do NOT return CR No, Patient Registration, Token No, Room No, Mobile, Fee amounts, or barcodes.
doctor_stamp: "present" if a stamp/seal is visible (even if text is faint), else "".
Use "" for doctor_registration_number only when no CN/Reg number appears near the stamp/signature."""

DOCUMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": ["handwritten", "computer_generated", "uncertain"],
        },
        "content_handwritten_percent": {"type": "number"},
        "content_computer_generated_percent": {"type": "number"},
        "is_medical_document": {"type": "boolean"},
        "non_medical_reason": {"type": "string"},
        "document_category": {
            "type": "string",
            "enum": ["prescription", "invoice", "report", "other"],
        },
        "invoice_subtype": {
            "type": "string",
            "enum": [
                "pharmacy",
                "diagnostic",
                "opd_consultation",
                "dental",
                "eye_care",
                "uncertain",
                "not_applicable",
            ],
        },
        "prescription_subtype": {
            "type": "string",
            "enum": [
                "opd",
                "pharmacy",
                "diagnostic",
                "dental",
                "eye_care",
                "uncertain",
                "not_applicable",
            ],
        },
        "prescription_parameters": PRESCRIPTION_PARAMS_SCHEMA,
        "invoice_parameters": INVOICE_PARAMS_SCHEMA,
        "report_parameters": REPORT_PARAMS_SCHEMA,
    },
    "required": [
        "document_type",
        "content_handwritten_percent",
        "content_computer_generated_percent",
        "is_medical_document",
        "non_medical_reason",
        "document_category",
        "invoice_subtype",
        "prescription_subtype",
        "prescription_parameters",
        "invoice_parameters",
        "report_parameters",
    ],
    "additionalProperties": False,
}

_VALID_CATEGORIES = frozenset({"prescription", "invoice", "report", "other"})


def _str_val(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _list_val(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    return [s for item in raw if (s := _str_val(item))]


def _normalize_medicine_item(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        return {
            "medicine": _str_val(raw.get("medicine")),
            "dosage": _str_val(raw.get("dosage")),
        }
    return {"medicine": _str_val(raw), "dosage": ""}


def _normalize_prescribed_medicines(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []
    items: List[Dict[str, str]] = []
    for entry in raw:
        item = _normalize_medicine_item(entry)
        if item["medicine"] or item["dosage"]:
            items.append(item)
    return items


def _normalize_invoice_detail_list(raw: Any) -> List[str]:
    """medicine_details / test_details as string arrays (coerce legacy object rows)."""
    if not isinstance(raw, list):
        return []
    items: List[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            name = _str_val(entry.get("name") or entry.get("product"))
            if name:
                items.append(name)
        else:
            line = _str_val(entry)
            if line:
                items.append(line)
    return items


def _normalize_params(
    raw: Any,
    keys: Sequence[str],
    array_keys: frozenset[str],
) -> Dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    result: Dict[str, Any] = {}
    for key in keys:
        if key == "prescribed_medicines":
            result[key] = _normalize_prescribed_medicines(data.get(key))
        elif key in _INVOICE_DETAIL_LIST_KEYS:
            result[key] = _normalize_invoice_detail_list(data.get(key))
        elif key in array_keys:
            result[key] = _list_val(data.get(key))
        else:
            result[key] = _str_val(data.get(key))
    return result


def _merge_string_field(target: Dict[str, Any], key: str, value: str) -> None:
    if _str_val(value) and not _str_val(target.get(key)):
        target[key] = _str_val(value)


def _normalize_doctor_registration(params: Dict[str, Any]) -> None:
    """Fill doctor_registration_number from stamp/name text; support CN No. labels."""
    reg = _str_val(params.get("doctor_registration_number"))
    if reg and reg.lower() != "present":
        cn = _CN_NO_RE.search(reg) or _DOCTOR_REG_IN_TEXT_RE.search(reg)
        if cn:
            candidate = cn.group(1).strip()
        else:
            candidate = reg
        # Reject patient CR No / long case IDs mistaken for doctor reg
        if _CR_NO_LIKE_RE.match(re.sub(r"[\s\-]", "", candidate)):
            params["doctor_registration_number"] = ""
        else:
            params["doctor_registration_number"] = candidate
        return

    stamp = _str_val(params.get("doctor_stamp"))
    sources = [
        stamp,
        _str_val(params.get("doctor_name")),
        _str_val(params.get("doctor_qualification")),
        _str_val(params.get("clinic_hospital_name")),
    ]
    for source in sources:
        if not source or source.lower() == "present":
            continue
        cn = _CN_NO_RE.search(source)
        if cn:
            candidate = cn.group(1).strip()
            if not _CR_NO_LIKE_RE.match(re.sub(r"[\s\-]", "", candidate)):
                params["doctor_registration_number"] = candidate
                return
        match = _DOCTOR_REG_IN_TEXT_RE.search(source)
        if match:
            candidate = match.group(1).strip()
            if not _CR_NO_LIKE_RE.match(re.sub(r"[\s\-]", "", candidate)):
                params["doctor_registration_number"] = candidate
                return
        digit_match = _DOCTOR_REG_DIGITS_RE.search(source)
        if digit_match and stamp and stamp.lower() != "present":
            candidate = digit_match.group(1)
            if not _CR_NO_LIKE_RE.match(candidate):
                params["doctor_registration_number"] = candidate
                return


def _is_meaningful_string(val: Any) -> bool:
    s = _str_val(val).lower()
    if not s:
        return False
    return s not in _PLACEHOLDER_VALUES


_GSTIN_PREFIX_RE = re.compile(
    r"(?<![A-Z0-9])(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]{2}[A-Z0-9])(?![A-Z0-9])",
    re.IGNORECASE,
)


def _extract_gstin(text: str) -> str:
    """Extract valid 15-char GSTIN. Also attempts to recover when position-13 Z is misread."""
    if not text:
        return ""
    # Asterisks often wrap GSTIN on govt pharmacy bills: **GSTIN:** **02AAH…**
    cleaned = re.sub(r"[*#]+", " ", text.upper())
    compact = re.sub(r"\s+", "", cleaned)
    compact = re.sub(r"GSTIN(?:/UIN)?[:/]?|GST\s*NO[./:]?", "|", compact)
    compact = re.sub(r"\|+", "|", compact)
    # Reject obvious FSSAI (long digit runs) before GSTIN search
    compact = re.sub(r"FSSAI(?:\s*NO)?[:/]?\d{8,}", "|", compact)

    match = _GSTIN_RE.search(compact)
    if match:
        return match.group(1).upper()

    # Label-anchored 15-char token (spaces/stars already stripped)
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


def _invoice_text_blob(inv: Dict[str, Any]) -> str:
    return " ".join(
        _str_val(inv.get(key))
        for key in (
            "provider_name",
            "provider_address",
            "invoice_number",
            "provider_contact",
            "gst_number",
            "drug_license_number",
        )
    ).lower()


def _looks_like_pharmacy_invoice(inv: Dict[str, Any], data: Dict[str, Any]) -> bool:
    subtype = str(data.get("invoice_subtype", ""))
    if subtype == "pharmacy":
        return True
    if _is_filled(inv, "medicine_details"):
        return True
    if _is_filled(inv, "drug_license_number") or _is_filled(inv, "gst_number"):
        return True
    blob = " ".join(
        _str_val(inv.get(key))
        for key in ("provider_name", "provider_address", "authorized_stamp")
    ).lower()
    return any(
        hint in blob
        for hint in (
            "medical store",
            "chemist",
            "pharmacy",
            "bill of supply",
            "pharmacist",
            "drug license",
            "dl no",
            "gst no",
        )
    )


def _looks_like_digital_or_pos_payment(inv: Dict[str, Any]) -> bool:
    payment = _str_val(inv.get("payment_mode")).lower()
    if payment and any(hint in payment for hint in _DIGITAL_PAYMENT_HINTS):
        return True
    return _is_filled(inv, "transaction_reference")


def _medicine_lines_look_printed(medicines: List[str]) -> bool:
    """Printed pharmacy bills: HSN codes, batch/expiry, or uniform typed product lines."""
    if not medicines:
        return False
    for line in medicines:
        if _PRINTED_MEDICINE_LINE_RE.search(line):
            return True
        if _HSN_CODE_LINE_RE.match(line.strip()):
            return True
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5 and any(parts[3:5]):
            return True
    if len(medicines) >= 3:
        typed = sum(1 for line in medicines if _TYPED_PRODUCT_RE.search(line))
        if typed >= len(medicines) - 1:
            return True
    return False


def _looks_like_handwritten_cash_memo(inv: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """Pre-printed cash memo / form with handwritten patient, items, amounts, signature."""
    if _looks_like_structured_printed_invoice(inv):
        return False
    if _looks_like_digital_or_pos_payment(inv):
        return False

    blob = " ".join(
        _str_val(inv.get(key))
        for key in ("provider_name", "provider_address", "authorized_stamp")
    ).lower()
    if "cash memo" in blob:
        return True

    medicines = _normalize_invoice_detail_list(inv.get("medicine_details"))
    sig = _str_val(inv.get("authorized_signature")).lower()
    has_hw_sig = sig == "present"

    if medicines and not _medicine_lines_look_printed(medicines):
        for line in medicines:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and not parts[-1] and (len(parts) < 5 or not parts[-2]):
                if has_hw_sig or _is_filled(inv, "patient_name"):
                    return True

    tests = _normalize_invoice_detail_list(inv.get("test_details"))
    if tests and not medicines and has_hw_sig and _is_filled(inv, "patient_name"):
        return True

    return False


def _looks_like_structured_printed_invoice(inv: Dict[str, Any]) -> bool:
    """Fully printed POS / tax invoice — not a handwritten cash memo on a pre-printed form."""
    has_line_items = _is_filled(inv, "medicine_details") or _is_filled(inv, "test_details")
    if not has_line_items:
        return False
    has_provider = _is_meaningful_string(inv.get("provider_name"))
    has_meta = _is_meaningful_string(inv.get("invoice_number")) and _is_meaningful_string(
        inv.get("invoice_date")
    )
    has_amount = _is_meaningful_string(inv.get("total_amount"))
    if not (has_provider and has_meta and has_amount):
        return False

    if _looks_like_digital_or_pos_payment(inv):
        return True

    blob = _invoice_text_blob(inv)
    if any(signal in blob for signal in _PRINTED_POS_INVOICE_SIGNALS):
        return True

    medicines = _normalize_invoice_detail_list(inv.get("medicine_details"))
    if medicines and _medicine_lines_look_printed(medicines):
        return True

    if len(medicines) >= 3:
        return True

    total = _str_val(inv.get("total_amount")).replace(",", "")
    if re.fullmatch(r"\d+\.\d{2}", total) and _is_filled(inv, "doctor_name"):
        return True

    inv_no = _str_val(inv.get("invoice_number"))
    if len(re.sub(r"\D", "", inv_no)) >= 8:
        return True

    return False


def _pharmacy_regulatory_invalid(inv: Dict[str, Any]) -> bool:
    gst = _extract_gstin(_str_val(inv.get("gst_number")))
    dl = _extract_drug_licenses(_str_val(inv.get("drug_license_number")))
    return not gst or not dl


def _pharmacy_gst_only_missing(inv: Dict[str, Any]) -> bool:
    """DL already extracted — only GSTIN still missing (skip full regulatory pass)."""
    if _extract_gstin(_str_val(inv.get("gst_number"))):
        return False
    return bool(_extract_drug_licenses(_str_val(inv.get("drug_license_number"))))


def _invoice_gst_missing(inv: Dict[str, Any]) -> bool:
    return not _extract_gstin(_str_val(inv.get("gst_number")))


_pharmacy_gst_missing = _invoice_gst_missing


def _pharmacy_regulatory_incomplete(inv: Dict[str, Any]) -> bool:
    return _pharmacy_regulatory_invalid(inv)


def _invoice_authorization_missing(inv: Dict[str, Any]) -> bool:
    """Stamp or signature at bill footer — either counts when visible."""
    return not _is_filled(inv, "authorized_stamp") and not _is_filled(
        inv, "authorized_signature"
    )





def _normalize_total_amount(params: Dict[str, Any]) -> None:
    """Strip labels from total_amount; keep numeric value only."""
    raw = _str_val(params.get("total_amount"))
    if not raw:
        return
    clean = raw.replace(",", "").strip()
    if re.fullmatch(r"\d+\.\d{2}", clean) or re.fullmatch(r"\d+", clean):
        params["total_amount"] = clean
        return
    labeled = re.search(
        r"(?:total(?:\s+mrp\s+value)?|grand\s+total|invoice\s+value|net\s+amount|amount)"
        r"\s*[:\s]*([0-9,]+(?:\.\d{1,2})?)",
        raw,
        re.IGNORECASE,
    )
    if labeled:
        params["total_amount"] = labeled.group(1).replace(",", "")
        return
    decimal = re.search(r"([0-9,]+\.\d{2})", raw)
    if decimal:
        params["total_amount"] = decimal.group(1).replace(",", "")


def _invoice_text_scan_blob(params: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in INVOICE_PARAM_KEYS:
        val = params.get(key)
        if key in _INVOICE_ARRAY_KEYS:
            parts.extend(_list_val(val))
        else:
            parts.append(_str_val(val))
    return " ".join(part for part in parts if part)


def _normalize_invoice_gst(params: Dict[str, Any]) -> None:
    """Extract GSTIN from gst_number field or any other extracted invoice text."""
    blob = _invoice_text_scan_blob(params)
    raw_gst = _str_val(params.get("gst_number"))
    gst = _extract_gstin(raw_gst) or _extract_gstin(blob)
    params["gst_number"] = gst


def _normalize_pharmacy_drug_license(params: Dict[str, Any]) -> None:
    """Normalize drug_license_number from field or scanned invoice text."""
    blob = _invoice_text_scan_blob(params)
    raw_dl = _str_val(params.get("drug_license_number"))
    licenses = _extract_drug_licenses(raw_dl) or _extract_drug_licenses(blob)
    params["drug_license_number"] = "; ".join(licenses) if licenses else ""


def _normalize_pharmacy_regulatory_ids(params: Dict[str, Any]) -> None:
    """Normalize gst_number and drug_license_number on pharmacy bills."""
    _normalize_invoice_gst(params)
    _normalize_pharmacy_drug_license(params)


def _apply_regulatory_from_refined(inv: Dict[str, Any], refined: Dict[str, Any]) -> None:
    """Prefer valid GSTIN / DL parsed from a refine pass; never clear existing valid values."""
    existing_gst = _extract_gstin(_str_val(inv.get("gst_number")))
    gst = _extract_gstin(_str_val(refined.get("gst_number")))
    if gst:
        inv["gst_number"] = gst
    elif existing_gst:
        inv["gst_number"] = existing_gst

    existing_dl = _extract_drug_licenses(_str_val(inv.get("drug_license_number")))
    raw_dl = _str_val(refined.get("drug_license_number"))
    licenses = _extract_drug_licenses(raw_dl)
    if licenses:
        inv["drug_license_number"] = "; ".join(licenses)
    elif existing_dl:
        inv["drug_license_number"] = "; ".join(existing_dl)
    elif _is_meaningful_string(raw_dl):
        inv["drug_license_number"] = raw_dl


def _normalize_invoice_fields(params: Dict[str, Any]) -> None:
    """Clear placeholders and normalize invoice string fields."""
    _normalize_invoice_placeholders(params)
    _normalize_total_amount(params)
    _normalize_invoice_gst(params)
    if _looks_like_pharmacy_invoice(params, {}):
        _normalize_pharmacy_drug_license(params)


def _normalize_invoice_placeholders(params: Dict[str, Any]) -> None:
    """Clear N/A-style placeholders from invoice string fields."""
    for key in INVOICE_PARAM_KEYS:
        if key in _INVOICE_ARRAY_KEYS:
            continue
        val = _str_val(params.get(key))
        if val and not _is_meaningful_string(val):
            params[key] = ""


def _is_filled(params: Dict[str, Any], key: str) -> bool:
    val = params.get(key)
    if key == "prescribed_medicines" and isinstance(val, list):
        return any(
            isinstance(item, dict) and _str_val(item.get("medicine")) for item in val
        )
    if isinstance(val, list):
        return len(val) > 0
    if not _is_meaningful_string(val):
        return False
    return True


def _completeness(
    params: Dict[str, Any],
    required: Sequence[str],
    extra_checks: Sequence[Tuple[str, bool]],
) -> Tuple[float, List[str]]:
    missing = [key for key in required if not _is_filled(params, key)]
    for label, ok in extra_checks:
        if not ok:
            missing.append(label)
    total = len(required) + len(extra_checks)
    filled = total - len(missing)
    pct = round(filled / total * 100.0, 2) if total else 0.0
    return pct, missing


def _provider_blob(params: Dict[str, Any]) -> str:
    parts = [
        _str_val(params.get("provider_name")),
        _str_val(params.get("provider_address")),
    ]
    for key in ("service_details", "item_details"):
        parts.extend(_list_val(params.get(key)))
    return " ".join(part for part in parts if part).lower()


def _infer_invoice_subtype(params: Dict[str, Any], raw_subtype: str) -> str:
    if raw_subtype in _VALID_INVOICE_SUBTYPES and raw_subtype not in (
        "uncertain",
        "not_applicable",
    ):
        return raw_subtype
    blob = _provider_blob(params)
    if _is_filled(params, "medicine_details") or _is_filled(params, "drug_license_number"):
        return "pharmacy"
    if _is_filled(params, "test_details"):
        return "diagnostic"
    if any(k in blob for k in ("optical", "optician", "eye care", "eyecare", "spectacle", "vision care")):
        return "eye_care"
    if any(k in blob for k in ("dental", "dentist", "orthodont", "tooth")):
        return "dental"
    if _is_filled(params, "item_details"):
        if any(k in blob for k in ("lens", "frame", "bifocal", "progressive", "contact lens")):
            return "eye_care"
        if "dental" in blob:
            return "dental"
    if (
        _is_filled(params, "service_details")
        or _is_filled(params, "consultation_charges")
        or _is_filled(params, "registration_charges")
    ):
        return "opd_consultation"
    if _is_filled(params, "doctor_name") and _is_filled(params, "total_amount"):
        return "opd_consultation"
    if _is_filled(params, "drug_license_number") or _is_filled(params, "gst_number"):
        return "pharmacy"
    return "uncertain"


def _invoice_authorization_present(inv: Dict[str, Any]) -> bool:
    return _is_filled(inv, "authorized_stamp") or _is_filled(inv, "authorized_signature")


def _invoice_has_line_items_or_charges(inv: Dict[str, Any]) -> bool:
    return (
        _is_filled(inv, "medicine_details")
        or _is_filled(inv, "test_details")
        or _is_filled(inv, "service_details")
        or _is_filled(inv, "item_details")
        or _is_filled(inv, "consultation_charges")
        or _is_filled(inv, "registration_charges")
    )


def _invoice_subtype_extra_checks(
    subtype: str, inv: Dict[str, Any]
) -> List[Tuple[str, bool]]:
    checks: List[Tuple[str, bool]] = [
        ("line_items_or_charges", _invoice_has_line_items_or_charges(inv)),
    ]
    if subtype == "pharmacy":
        checks.extend(
            (
                ("gst_number", bool(_extract_gstin(_str_val(inv.get("gst_number"))))),
                (
                    "drug_license_number",
                    bool(_extract_drug_licenses(_str_val(inv.get("drug_license_number")))),
                ),
                ("medicine_details", _is_filled(inv, "medicine_details")),
                ("authorization", _invoice_authorization_present(inv)),
                ("payment_mode", _is_filled(inv, "payment_mode")),
            )
        )
    elif subtype == "opd_consultation":
        checks.extend(
            (
                ("doctor_name", _is_filled(inv, "doctor_name")),
                ("provider_address", _is_filled(inv, "provider_address")),
                ("payment_mode", _is_filled(inv, "payment_mode")),
                ("authorization", _invoice_authorization_present(inv)),
            )
        )
    elif subtype == "diagnostic":
        checks.extend(
            (
                ("test_details", _is_filled(inv, "test_details")),
                ("provider_address", _is_filled(inv, "provider_address")),
                ("authorization", _invoice_authorization_present(inv)),
            )
        )
    elif subtype == "dental":
        checks.extend(
            (
                (
                    "item_or_service_details",
                    _is_filled(inv, "item_details") or _is_filled(inv, "service_details"),
                ),
                ("provider_address", _is_filled(inv, "provider_address")),
                ("authorization", _invoice_authorization_present(inv)),
                ("payment_mode", _is_filled(inv, "payment_mode")),
            )
        )
    elif subtype == "eye_care":
        checks.extend(
            (
                ("item_details", _is_filled(inv, "item_details")),
                ("provider_address", _is_filled(inv, "provider_address")),
                ("authorization", _invoice_authorization_present(inv)),
                ("payment_mode", _is_filled(inv, "payment_mode")),
            )
        )
    return checks


def _prescription_blob(params: Dict[str, Any]) -> str:
    parts = [
        _str_val(params.get("clinic_hospital_name")),
        _str_val(params.get("clinic_hospital_address")),
        _str_val(params.get("doctor_qualification")),
        _str_val(params.get("diagnosis")),
        _str_val(params.get("line_of_treatment")),
        _str_val(params.get("treatment_advised")),
        _str_val(params.get("procedure_recommendation")),
        _str_val(params.get("treatment_advice")),
    ]
    return " ".join(part for part in parts if part).lower()


def _normalize_prescription_fields(params: Dict[str, Any]) -> None:
    if not _is_filled(params, "presenting_complaints"):
        complaints = _str_val(params.get("complaints"))
        if complaints:
            params["presenting_complaints"] = complaints
    params.pop("complaints", None)
    if not _is_filled(params, "line_of_treatment"):
        plan = _str_val(params.get("treatment_plan"))
        if plan:
            params["line_of_treatment"] = plan


def _infer_prescription_subtype(params: Dict[str, Any], raw_subtype: str) -> str:
    if raw_subtype in _VALID_PRESCRIPTION_SUBTYPES and raw_subtype not in (
        "uncertain",
        "not_applicable",
    ):
        return raw_subtype
    blob = _prescription_blob(params)
    if any(
        k in blob
        for k in ("ophthalm", "optometr", "eye care", "eyecare", "visual acuity", "spectacle")
    ):
        return "eye_care"
    if (
        _is_filled(params, "eye_power_prescription")
        or _is_filled(params, "visual_acuity_details")
        or _is_filled(params, "glasses_contact_lens_prescription")
    ):
        return "eye_care"
    if any(k in blob for k in ("dental", "dentist", "orthodont", "tooth")):
        return "dental"
    if (
        _is_filled(params, "affected_tooth_number")
        or _is_filled(params, "procedure_recommendation")
        or _is_filled(params, "treatment_advised")
    ):
        return "dental"
    has_tests = _is_filled(params, "advised_tests")
    has_meds = _is_filled(params, "prescribed_medicines")
    if has_tests and not has_meds:
        return "diagnostic"
    if raw_subtype == "pharmacy":
        return "pharmacy"
    if has_meds:
        return "opd"
    if has_tests:
        return "diagnostic"
    return "uncertain"


def _prescription_subtype_extra_checks(
    subtype: str, rx: Dict[str, Any]
) -> List[Tuple[str, bool]]:
    checks: List[Tuple[str, bool]] = [
        ("patient_age", _is_filled(rx, "patient_age")),
        ("patient_gender", _is_filled(rx, "patient_gender")),
        (
            "diagnosis_or_presenting_complaints",
            _is_filled(rx, "diagnosis") or _is_filled(rx, "presenting_complaints"),
        ),
    ]
    if subtype == "opd":
        checks.extend(
            (
                ("line_of_treatment", _is_filled(rx, "line_of_treatment")),
                (
                    "prescribed_medicines_or_advised_tests",
                    _is_filled(rx, "prescribed_medicines")
                    or _is_filled(rx, "advised_tests"),
                ),
            )
        )
    elif subtype == "pharmacy":
        checks.extend(
            (
                ("prescribed_medicines", _is_filled(rx, "prescribed_medicines")),
                ("clinic_hospital_address", _is_filled(rx, "clinic_hospital_address")),
            )
        )
    elif subtype == "diagnostic":
        checks.extend(
            (
                ("line_of_treatment", _is_filled(rx, "line_of_treatment")),
                ("advised_tests", _is_filled(rx, "advised_tests")),
            )
        )
    elif subtype == "dental":
        checks.extend(
            (
                (
                    "treatment_advised_or_procedure",
                    _is_filled(rx, "treatment_advised")
                    or _is_filled(rx, "procedure_recommendation"),
                ),
                ("treatment_plan", _is_filled(rx, "treatment_plan")),
            )
        )
    elif subtype == "eye_care":
        checks.extend(
            (
                (
                    "eye_power_or_visual_acuity",
                    _is_filled(rx, "eye_power_prescription")
                    or _is_filled(rx, "visual_acuity_details"),
                ),
                ("treatment_advice", _is_filled(rx, "treatment_advice")),
            )
        )
    return checks


def _correct_document_category(
    doc_type: str,
    category: str,
    invoice_params: Dict[str, Any],
) -> str:
    if category != "invoice":
        return category
    if not _is_filled(invoice_params, "total_amount"):
        if doc_type == "handwritten":
            return "prescription"
        if doc_type == "computer_generated" and not (
            _is_filled(invoice_params, "test_details")
            or _is_filled(invoice_params, "medicine_details")
        ):
            return "report"
    return category


def _content_percent_split(data: Dict[str, Any]) -> Tuple[float, float]:
    """Handwriting split for filled content only."""
    hw = max(0.0, min(100.0, float(data.get("content_handwritten_percent", 50.0))))
    cg = max(0.0, min(100.0, float(data.get("content_computer_generated_percent", 50.0))))
    total = hw + cg
    if total < 1e-6:
        return 50.0, 50.0
    hw = round(hw / total * 100.0, 2)
    return hw, round(100.0 - hw, 2)


def _is_generic_invoice_service_line(line: str) -> bool:
    """Printed bill pad row label (category row), not a billed test/medicine line."""
    text = line.strip()
    if not text:
        return True
    if _INVOICE_PRINTED_ROW_RE.search(text):
        return True
    low = text.lower()
    # Numbered category row without price in the same string
    if re.match(r"^\d+[\).:\s-]", text) and "examination" in low and "rs" not in low:
        return True
    return False


def _fix_content_classification(data: Dict[str, Any], inv_params: Dict[str, Any]) -> None:
    """Cash memo / handwritten fill mislabelled as computer_generated."""
    if _looks_like_handwritten_cash_memo(inv_params, data):
        return
    if _looks_like_structured_printed_invoice(inv_params):
        return
    doc_type = str(data.get("document_type", "uncertain"))
    tests = _normalize_invoice_detail_list(inv_params.get("test_details"))
    services = _list_val(inv_params.get("service_details"))
    generic_only = bool(services) and all(_is_generic_invoice_service_line(s) for s in services)
    hw = float(data.get("content_handwritten_percent", 0))

    if doc_type == "computer_generated" and hw < 15 and (generic_only or tests):
        data["document_type"] = "handwritten"
        data["content_handwritten_percent"] = 100.0
        data["content_computer_generated_percent"] = 0.0


def _fix_handwritten_invoice_classification(
    data: Dict[str, Any], inv_params: Dict[str, Any]
) -> None:
    """Pre-printed form with handwritten fill (cash memo) → document_type=handwritten."""
    if not _looks_like_handwritten_cash_memo(inv_params, data):
        return
    hw_pct = 92.0 if _str_val(inv_params.get("authorized_signature")) == "present" else 88.0
    data["document_type"] = "handwritten"
    data["content_handwritten_percent"] = hw_pct
    data["content_computer_generated_percent"] = round(100.0 - hw_pct, 2)


def _fix_pharmacy_invoice_content_classification(
    data: Dict[str, Any], inv_params: Dict[str, Any]
) -> None:
    """Printed structured invoices mislabelled as handwritten → computer_generated."""
    if _looks_like_handwritten_cash_memo(inv_params, data):
        return
    if not _looks_like_structured_printed_invoice(inv_params):
        return

    doc_type = str(data.get("document_type", "uncertain"))
    hw = float(data.get("content_handwritten_percent", 0))
    if doc_type == "computer_generated" and hw < 20:
        return
    if doc_type not in ("handwritten", "uncertain") and hw < 50:
        return

    sig = _str_val(inv_params.get("authorized_signature"))
    stamp = _str_val(inv_params.get("authorized_stamp"))
    hw_content = 10.0 if sig == "present" or stamp == "present" else 5.0

    data["document_type"] = "computer_generated"
    data["content_handwritten_percent"] = hw_content
    data["content_computer_generated_percent"] = round(100.0 - hw_content, 2)


def _looks_like_typed_lab_report(report_params: Dict[str, Any]) -> bool:
    """Printed lab report: numeric results + reference ranges + patient name present."""
    if not _is_filled(report_params, "test_results"):
        return False
    if not _is_filled(report_params, "reference_ranges"):
        return False
    if not _is_filled(report_params, "patient_name"):
        return False
    for result in _list_val(report_params.get("test_results")):
        if any(ch.isdigit() for ch in result):
            return True
    return False


def _fix_report_content_classification(
    data: Dict[str, Any], report_params: Dict[str, Any]
) -> None:
    """Typed lab reports mislabelled as handwritten → computer_generated content."""
    if str(data.get("document_category")) != "report":
        return
    if not _looks_like_typed_lab_report(report_params):
        return

    doc_type = str(data.get("document_type", "uncertain"))
    hw = float(data.get("content_handwritten_percent", 0))
    if doc_type != "handwritten" and hw < 50:
        return

    sig = _str_val(report_params.get("pathologist_signature"))
    hw_content = 10.0 if sig == "present" else 5.0

    data["document_type"] = "computer_generated"
    data["content_handwritten_percent"] = hw_content
    data["content_computer_generated_percent"] = round(100.0 - hw_content, 2)


def _apply_document_type(data: Dict[str, Any]) -> str:
    doc_type = str(data.get("document_type", "uncertain"))
    if doc_type not in ("handwritten", "computer_generated", "uncertain"):
        return "uncertain"
    return doc_type


def _count_extracted_fields(params: Dict[str, Any]) -> int:
    count = 0
    for key, val in params.items():
        if key == "prescribed_medicines" and isinstance(val, list):
            count += sum(
                1
                for item in val
                if isinstance(item, dict)
                and (_str_val(item.get("medicine")) or _str_val(item.get("dosage")))
            )
        elif isinstance(val, list):
            count += len(val)
        elif _str_val(val):
            count += 1
    return count


def _non_medical_reason_suggests_bill(data: Dict[str, Any]) -> bool:
    reason = _str_val(data.get("non_medical_reason")).lower()
    if not reason:
        return False
    return (
        any(hint in reason for hint in _MEDICAL_CLAIM_HINTS)
        or "bill" in reason
        or "invoice" in reason
        or "receipt" in reason
    )


def _looks_like_opd_consultation_invoice(inv: Dict[str, Any]) -> bool:
    if not _is_filled(inv, "total_amount") or not _is_filled(inv, "patient_name"):
        return False
    return (
        _is_filled(inv, "provider_name")
        or _is_filled(inv, "doctor_name")
        or _is_filled(inv, "service_details")
        or _is_filled(inv, "invoice_number")
    )



def _invoice_extraction_score(inv: Dict[str, Any]) -> int:
    score = 0
    if _is_filled(inv, "patient_name"):
        score += 2
    if _is_filled(inv, "total_amount"):
        score += 2
    if _is_filled(inv, "invoice_number"):
        score += 1
    if _is_filled(inv, "medicine_details"):
        score += 2
    if _is_filled(inv, "test_details"):
        score += 2
    if _is_filled(inv, "gst_number") or _is_filled(inv, "drug_license_number"):
        score += 1
    if _is_filled(inv, "provider_name"):
        score += 1
    return score


def _recover_medical_classification(data: Dict[str, Any]) -> None:
    """Undo false 'other' when extraction or reason indicates a real bill/Rx/report."""
    inv = _normalize_params(
        data.get("invoice_parameters"), INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS
    )
    rx = _normalize_params(
        data.get("prescription_parameters"),
        PRESCRIPTION_PARAM_KEYS,
        _PRESCRIPTION_ARRAY_KEYS,
    )
    rep = _normalize_params(
        data.get("report_parameters"), REPORT_PARAM_KEYS, _REPORT_ARRAY_KEYS
    )

    inv_score = _invoice_extraction_score(inv)
    rx_score = _count_extracted_fields(rx)
    rep_score = _count_extracted_fields(rep)

    if inv_score >= 3 or (inv_score >= 2 and _is_filled(inv, "medicine_details")):
        data["is_medical_document"] = True
        data["document_category"] = "invoice"
        data["non_medical_reason"] = ""
        if _is_filled(inv, "medicine_details") or _is_filled(inv, "drug_license_number"):
            data["invoice_subtype"] = "pharmacy"
        elif _looks_like_opd_consultation_invoice(inv):
            data["invoice_subtype"] = "opd_consultation"
        return

    if inv_score >= 2 and _looks_like_opd_consultation_invoice(inv):
        data["is_medical_document"] = True
        data["document_category"] = "invoice"
        data["invoice_subtype"] = "opd_consultation"
        data["non_medical_reason"] = ""
        return

    if rx_score >= 3:
        data["is_medical_document"] = True
        data["document_category"] = "prescription"
        data["invoice_subtype"] = "not_applicable"
        data["prescription_subtype"] = _infer_prescription_subtype(
            rx, str(data.get("prescription_subtype", "uncertain"))
        )
        data["non_medical_reason"] = ""
        return

    if rep_score >= 3:
        data["is_medical_document"] = True
        data["document_category"] = "report"
        data["invoice_subtype"] = "not_applicable"
        data["non_medical_reason"] = ""
        return

    if _non_medical_reason_suggests_bill(data):
        data["is_medical_document"] = True
        data["document_category"] = "invoice"
        reason = _str_val(data.get("non_medical_reason")).lower()
        if any(h in reason for h in ("consultation", "clinic", "opd", "doctor", "physician")):
            data["invoice_subtype"] = "opd_consultation"
        else:
            data["invoice_subtype"] = "pharmacy"
        data["non_medical_reason"] = ""


def _all_medical_blocks_empty(data: Dict[str, Any]) -> bool:
    rx = _normalize_params(
        data.get("prescription_parameters"),
        PRESCRIPTION_PARAM_KEYS,
        _PRESCRIPTION_ARRAY_KEYS,
    )
    inv = _normalize_params(
        data.get("invoice_parameters"), INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS
    )
    rep = _normalize_params(
        data.get("report_parameters"), REPORT_PARAM_KEYS, _REPORT_ARRAY_KEYS
    )
    return (
        _count_extracted_fields(rx) == 0
        and _count_extracted_fields(inv) == 0
        and _count_extracted_fields(rep) == 0
    )


def _resolve_non_medical(
    data: Dict[str, Any], category: str
) -> Tuple[str, str, bool]:
    """Return (category, reason, is_medical)."""
    reason = _str_val(data.get("non_medical_reason"))
    is_medical = bool(data.get("is_medical_document", True))

    if category == "other":
        return "other", reason or "Not a medical claim document", False

    if not is_medical and not _non_medical_reason_suggests_bill(data):
        return (
            "other",
            reason or "Image is not a prescription, invoice, or lab report",
            False,
        )

    if _all_medical_blocks_empty(data) and not _non_medical_reason_suggests_bill(data):
        return (
            "other",
            reason or "No prescription, bill, or report content found in the image",
            False,
        )

    return category, "", True


def _build_public_response(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    _recover_medical_classification(data)

    category = str(data.get("document_category", "other"))
    if category not in _VALID_CATEGORIES:
        category = "other"

    inv_params = _normalize_params(
        data.get("invoice_parameters"), INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS
    )
    if category == "invoice":
        _fix_content_classification(data, inv_params)
        _normalize_invoice_fields(inv_params)
        _fix_pharmacy_invoice_content_classification(data, inv_params)
        _fix_handwritten_invoice_classification(data, inv_params)
        data["invoice_parameters"] = inv_params
        inv_params = _normalize_params(
            data.get("invoice_parameters"), INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS
        )
    doc_type = _apply_document_type(data)
    content_hw, content_cg = _content_percent_split(data)
    category = _correct_document_category(doc_type, category, inv_params)
    category, non_medical_reason, is_medical = _resolve_non_medical(data, category)

    if not is_medical:
        return {
            "url": url,
            "is_medical_document": False,
            "document_type": doc_type,
            "document_category": "other",
            "invoice_subtype": "not_applicable",
            "prescription_subtype": "not_applicable",
            "handwritten_percent": content_hw,
            "computer_generated_percent": content_cg,
            "completeness_percent": 0.0,
            "parameters": _empty_all_claim_parameters(),
            "missing_parameters": ["not_a_medical_document"],
            "message": non_medical_reason,
        }

    prescription_subtype = "not_applicable"
    if category == "prescription":
        parameters = _normalize_params(
            data.get("prescription_parameters"),
            PRESCRIPTION_PARAM_KEYS,
            _PRESCRIPTION_ARRAY_KEYS,
        )
        _normalize_prescription_fields(parameters)
        _normalize_doctor_registration(parameters)
        prescription_subtype = _infer_prescription_subtype(
            parameters, str(data.get("prescription_subtype", "uncertain"))
        )
        # Keep full Rx extraction for the unified schema (not subtype-filtered).
        required = PRESCRIPTION_SUBTYPE_REQUIRED.get(
            prescription_subtype, PRESCRIPTION_REQUIRED
        )
        extra_checks = _prescription_subtype_extra_checks(prescription_subtype, parameters)
        completeness, missing_parameters = _completeness(
            parameters,
            required,
            tuple(extra_checks),
        )
        invoice_subtype = "not_applicable"
    elif category == "invoice":
        parameters = inv_params
        _normalize_invoice_fields(parameters)
        invoice_subtype = _infer_invoice_subtype(
            parameters, str(data.get("invoice_subtype", "uncertain"))
        )
        extra_checks = _invoice_subtype_extra_checks(invoice_subtype, parameters)
        completeness, missing_parameters = _completeness(
            parameters,
            INVOICE_REQUIRED,
            tuple(extra_checks),
        )
    else:
        parameters = _normalize_params(
            data.get("report_parameters"), REPORT_PARAM_KEYS, _REPORT_ARRAY_KEYS
        )
        _fix_report_content_classification(data, parameters)
        parameters = _normalize_params(
            data.get("report_parameters"), REPORT_PARAM_KEYS, _REPORT_ARRAY_KEYS
        )
        doc_type = _apply_document_type(data)
        content_hw, content_cg = _content_percent_split(data)
        completeness, missing_parameters = _completeness(parameters, REPORT_REQUIRED, ())
        invoice_subtype = "not_applicable"

    return {
        "url": url,
        "is_medical_document": True,
        "document_type": doc_type,
        "document_category": category,
        "invoice_subtype": invoice_subtype,
        "prescription_subtype": prescription_subtype,
        "handwritten_percent": content_hw,
        "computer_generated_percent": content_cg,
        "completeness_percent": completeness,
        "parameters": _to_all_claim_parameters(parameters),
        "missing_parameters": missing_parameters,
        "message": "",
    }








def _peek_pharmacy_regulatory_needs_refine(data: Dict[str, Any]) -> bool:
    """Dedicated header pass when GST or DL still missing on a pharmacy bill."""
    inv_raw = data.get("invoice_parameters")
    inv = dict(inv_raw) if isinstance(inv_raw, dict) else {}
    if not _looks_like_pharmacy_invoice(inv, data):
        return False
    _normalize_invoice_fields(inv)
    return _pharmacy_regulatory_invalid(inv)


def _peek_invoice_gst_needs_refine(data: Dict[str, Any]) -> bool:
    """Header GST pass for any invoice type when GSTIN not yet extracted."""
    if str(data.get("document_category")) != "invoice":
        return False
    if not data.get("is_medical_document", True):
        return False
    inv_raw = data.get("invoice_parameters")
    inv = dict(inv_raw) if isinstance(inv_raw, dict) else {}
    _normalize_invoice_fields(inv)
    return _invoice_gst_missing(inv)


def _peek_doctor_registration_needs_refine(data: Dict[str, Any]) -> bool:
    """Stamp-zone pass when Rx has empty/illegible doctor registration (e.g. CN No)."""
    if str(data.get("document_category")) != "prescription":
        return False
    if not data.get("is_medical_document", True):
        return False
    rx_raw = data.get("prescription_parameters")
    if not isinstance(rx_raw, dict):
        return False
    reg = _str_val(rx_raw.get("doctor_registration_number"))
    return not reg or reg.lower() == "present"


def _retry_wait_seconds(error: Exception, attempt: int) -> float:
    """Parse OpenAI 'try again in Xms' or use exponential backoff."""
    message = str(error)
    match_ms = re.search(r"try again in (\d+(?:\.\d+)?)\s*ms", message, re.IGNORECASE)
    if match_ms:
        return float(match_ms.group(1)) / 1000.0 + 0.05
    match_s = re.search(r"try again in (\d+(?:\.\d+)?)\s*s", message, re.IGNORECASE)
    if match_s:
        return float(match_s.group(1)) + 0.05
    return min(60.0, 0.5 * (2**attempt))


def _is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, APIStatusError) and error.status_code == 429:
        return True
    return "rate_limit" in str(error).lower() or "429" in str(error)


def _call_openai_json(
    client: Any,
    model: str,
    system: str,
    user_text: str,
    image_blocks: List[Dict[str, Any]],
    schema_name: str,
    schema: Dict[str, Any],
    max_tokens: int,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}, *image_blocks],
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                temperature=0.0,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("OpenAI response is not a JSON object")
            return parsed
        except Exception as exc:
            last_error = exc
            if not _is_rate_limit_error(exc) or attempt >= OPENAI_MAX_RETRIES - 1:
                raise
            wait = _retry_wait_seconds(exc, attempt)
            logger.warning(
                "OpenAI rate limit on %s (attempt %s/%s), retrying in %.2fs",
                schema_name,
                attempt + 1,
                OPENAI_MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI call failed without error detail")


def _pause_between_openai_calls() -> None:
    if OPENAI_INTER_CALL_DELAY_MS > 0:
        time.sleep(OPENAI_INTER_CALL_DELAY_MS / 1000.0)


def _call_openai_vision(
    client: Any,
    model: str,
    image_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    page_note = (
        " Multiple pages attached — read all pages and merge extracted fields."
        if len(image_blocks) > 1
        else ""
    )
    return _call_openai_json(
        client,
        model,
        SYSTEM_PROMPT,
        (
            "Classify content type and extract parameters."
            + page_note
            + " Use document_category=other for app screenshots or non-medical images."
            + " Fill only the matching parameter object."
        ),
        image_blocks,
        "medical_document_extraction",
        DOCUMENT_SCHEMA,
        2200,
    )






def _apply_doctor_reg_from_refined(
    rx: Dict[str, Any], refined: Dict[str, Any]
) -> None:
    reg = _str_val(refined.get("doctor_registration_number"))
    if reg and reg.lower() != "present" and not _str_val(rx.get("doctor_registration_number")):
        rx["doctor_registration_number"] = reg
    elif reg and reg.lower() != "present":
        existing = _str_val(rx.get("doctor_registration_number"))
        if not existing or existing.lower() == "present":
            rx["doctor_registration_number"] = reg
    stamp = _str_val(refined.get("doctor_stamp"))
    if stamp and not _str_val(rx.get("doctor_stamp")):
        rx["doctor_stamp"] = stamp


def _refine_doctor_registration_stamp(
    client: Any,
    model: str,
    data: Dict[str, Any],
    doc: DocumentPages,
) -> None:
    """Vision pass on bottom/stamp crops when doctor_registration_number is still empty."""
    crop_blocks = build_stamp_blocks_from_document(doc.raw, doc=doc)
    if not crop_blocks:
        return

    refined = _call_openai_json(
        client,
        model,
        DOCTOR_REG_PROMPT,
        (
            "Zoomed bottom / signature-zone crops of a prescription or OPD card. "
            "Find CN No / Reg No on or under the doctor stamp or signature. "
            "Ignore CR No / Token / Mobile."
        ),
        crop_blocks,
        "doctor_registration_stamp_extraction",
        DOCTOR_REG_SCHEMA,
        250,
    )
    rx_raw = data.get("prescription_parameters")
    rx: Dict[str, Any] = dict(rx_raw) if isinstance(rx_raw, dict) else {}
    _apply_doctor_reg_from_refined(rx, refined)
    _normalize_doctor_registration(rx)
    data["prescription_parameters"] = rx


def _refine_invoice_gst_header(
    client: Any,
    model: str,
    document_raw: bytes,
    data: Dict[str, Any],
    doc: DocumentPages | None = None,
) -> None:
    """GST-only pass on upscaled header crops when GSTIN is still missing on any invoice."""
    crop_blocks = build_gst_header_blocks_from_document(document_raw, doc=doc)
    if not crop_blocks:
        return

    inv_model = (os.getenv("OPENAI_INVOICE_MODEL") or model).strip()
    refined = _call_openai_json(
        client,
        inv_model,
        INVOICE_GST_PROMPT,
        (
            "Zoomed top header crops of a medical invoice. GSTIN is often small text in the "
            "top-right or top margin (near Regd. No., shop name, or letterhead). "
            "Extract gst_number: the 15-character GSTIN. Return the code only with no label. "
            "Use \"\" only if no GST number is printed on the document."
        ),
        crop_blocks,
        "invoice_gst_header_extraction",
        INVOICE_GST_SCHEMA,
        400,
    )
    inv_raw = data.get("invoice_parameters")
    inv: Dict[str, Any] = dict(inv_raw) if isinstance(inv_raw, dict) else {}
    _apply_regulatory_from_refined(inv, refined)
    _normalize_invoice_gst(inv)
    if _looks_like_pharmacy_invoice(inv, data):
        _normalize_pharmacy_drug_license(inv)
    data["invoice_parameters"] = inv


def _refine_gst_dl_if_needed(
    client: Any,
    model: str,
    image_blocks: List[Dict[str, Any]],
    data: Dict[str, Any],
    doc: DocumentPages,
) -> None:
    """Shared Vision fallback when Textract / main pass left GST or DL empty."""
    if _peek_pharmacy_regulatory_needs_refine(data):
        inv_raw = data.get("invoice_parameters")
        inv = dict(inv_raw) if isinstance(inv_raw, dict) else {}
        _normalize_invoice_fields(inv)
        if _pharmacy_gst_only_missing(inv):
            _refine_invoice_gst_header(client, model, doc.raw, data, doc)
        else:
            _refine_pharmacy_regulatory(
                client, model, image_blocks, data, doc.raw, doc
            )
    elif _peek_invoice_gst_needs_refine(data):
        _refine_invoice_gst_header(client, model, doc.raw, data, doc)


def _refine_pharmacy_regulatory(
    client: Any,
    model: str,
    image_blocks: List[Dict[str, Any]],
    data: Dict[str, Any],
    document_raw: bytes | None = None,
    doc: DocumentPages | None = None,
) -> None:
    """Single pass for top-header GSTIN and drug license (no retry — saves ~15–30s)."""
    raw = document_raw or (doc.raw if doc else b"")
    inv_model = (os.getenv("OPENAI_INVOICE_MODEL") or model).strip()
    refine_blocks = build_regulatory_header_blocks(image_blocks, raw, doc=doc)
    has_crop = bool(build_header_crop_from_document(raw, doc=doc))
    user_text = (
        "Extract gst_number (15-char GSTIN, no label) and drug_license_number "
        "(every DL NO. line, joined with '; ') from the bill header."
    )
    if has_crop:
        user_text += (
            " Multiple images: full bill plus upscaled top header crops — GST NO. is often "
            "ONLY visible in the top-right margin in small font near Regd. No. / Licence No."
        )

    refined = _call_openai_json(
        client,
        inv_model,
        PHARMACY_REGULATORY_PROMPT,
        user_text,
        refine_blocks,
        "pharmacy_regulatory_extraction",
        PHARMACY_REGULATORY_SCHEMA,
        350,
    )
    inv_raw = data.get("invoice_parameters")
    inv: Dict[str, Any] = dict(inv_raw) if isinstance(inv_raw, dict) else {}
    _apply_regulatory_from_refined(inv, refined)
    _normalize_pharmacy_regulatory_ids(inv)
    data["invoice_parameters"] = inv

    if raw and _invoice_gst_missing(inv):
        _refine_invoice_gst_header(client, model, raw, data, doc=doc)



def _run_textract_ocr(doc: DocumentPages) -> Dict[str, str]:
    if not textract_enabled():
        return {}
    try:
        return extract_textract_fields(doc.raw, page_images=doc.page_images)
    except Exception:
        logger.exception("Textract OCR failed")
        return {}


def classify_document_url_openai(url: str) -> Dict[str, Any]:
    """Hybrid classify: Textract OCR + OpenAI Vision in parallel, then merge.

    Optional Vision GST/DL header crop only when regulatory fields are still missing.
    """
    model = (os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    client = get_openai_client()
    doc = load_document(url)
    image_blocks = build_vision_blocks_from_document(doc)

    ocr: Dict[str, str] = {}
    if textract_enabled():
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_ocr = pool.submit(_run_textract_ocr, doc)
            fut_ai = pool.submit(_call_openai_vision, client, model, image_blocks)
            ocr = fut_ocr.result() or {}
            data = fut_ai.result()
    else:
        data = _call_openai_vision(client, model, image_blocks)

    _recover_medical_classification(data)
    if ocr:
        merge_textract_into_openai_data(data, ocr)
        _recover_medical_classification(data)

    try:
        if _peek_pharmacy_regulatory_needs_refine(
            data
        ) or _peek_invoice_gst_needs_refine(data):
            _pause_between_openai_calls()
            _refine_gst_dl_if_needed(client, model, image_blocks, data, doc)
    except Exception:
        logger.exception("GST/DL header pass failed for %s", url)

    try:
        if _peek_doctor_registration_needs_refine(data):
            _pause_between_openai_calls()
            _refine_doctor_registration_stamp(client, model, data, doc)
    except Exception:
        logger.exception("Doctor registration stamp pass failed for %s", url)

    if ocr:
        merge_textract_into_openai_data(data, ocr)

    return _build_public_response(url, data)
