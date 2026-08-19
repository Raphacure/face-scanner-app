"""
Classify medical / claim document images and extract structured parameters
for prescription (Rx), invoice, diagnostic report, and payment receipt.
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
    extract_pdf_text,
    load_document,
    pdf_vision_max_pages,
)
from app.services.textract_ocr import (
    extract_textract_fields,
    merge_textract_into_openai_data,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_RETRIES = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "6")))
OPENAI_INTER_CALL_DELAY_MS = max(0, int(os.getenv("OPENAI_INTER_CALL_DELAY_MS", "0")))
# Extra Vision refine passes (GST crop / doctor stamp) — off by default for latency.
# Set OPENAI_REFINE_PASSES=true only when first-pass + Textract miss GST/CN.
_OPENAI_REFINE_PASSES = (os.getenv("OPENAI_REFINE_PASSES") or "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

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

# Payment proof / UPI / bank transfer / card confirmation screenshots.
PAYMENT_RECEIPT_PARAM_KEYS: Tuple[str, ...] = (
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

PAYMENT_RECEIPT_REQUIRED: Tuple[str, ...] = (
    "payment_amount",
    "transaction_date",
    "payment_status",
    "payment_mode",
)


def _unique_param_keys(*groups: Sequence[str]) -> Tuple[str, ...]:
    """Stable union of param keys across prescription / invoice / report / payment."""
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
    PAYMENT_RECEIPT_PARAM_KEYS,
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


def _to_all_claim_parameters(partial: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay category extraction onto the full claim field schema."""
    return _normalize_params(partial, ALL_CLAIM_PARAM_KEYS, _ALL_CLAIM_ARRAY_KEYS)


def _requested_array_keys(fields: Sequence[str]) -> frozenset[str]:
    return frozenset(
        key
        for key in fields
        if key in _ALL_CLAIM_ARRAY_KEYS or key == "prescribed_medicines"
    )


def _to_requested_parameters(
    partial: Dict[str, Any],
    extract_fields: Sequence[str],
) -> Dict[str, Any]:
    """Return only CRM-requested keys (slim response)."""
    keys = tuple(extract_fields)
    return _normalize_params(partial, keys, _requested_array_keys(keys))


def _finalize_parameters(
    partial: Dict[str, Any],
    extract_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if extract_fields:
        return _to_requested_parameters(partial, extract_fields)
    return _to_all_claim_parameters(partial)


def _parameters_from_extraction(
    data: Dict[str, Any],
    extract_fields: Sequence[str],
) -> Dict[str, Any]:
    """Build the public parameters map from model output using the API field list."""
    partial: Dict[str, Any] = {}
    array_keys = _requested_array_keys(extract_fields)
    sources = (
        data.get("parameters"),
        data.get("prescription_parameters"),
        data.get("invoice_parameters"),
        data.get("report_parameters"),
        data.get("payment_receipt_parameters"),
    )
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in extract_fields:
            raw = source.get(key)
            if key == "prescribed_medicines":
                if not partial.get(key) and isinstance(raw, list) and raw:
                    partial[key] = raw
                continue
            if key in array_keys:
                if not partial.get(key) and isinstance(raw, list) and raw:
                    partial[key] = raw
                continue
            if _is_meaningful_string(raw) and not _is_meaningful_string(partial.get(key)):
                partial[key] = _str_val(raw)
    params = _to_requested_parameters(partial, extract_fields)
    _normalize_report_fields(params)
    _normalize_doctor_name_value(params)
    _normalize_doctor_registration(params)
    return params


def _merge_claim_field_buckets(*buckets: Dict[str, Any]) -> Dict[str, Any]:
    """Union non-empty fields across Rx / invoice / report / payment buckets (first wins)."""
    merged: Dict[str, Any] = {}
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        for key, raw in bucket.items():
            if key not in ALL_CLAIM_PARAM_KEYS:
                continue
            if key in _ALL_CLAIM_ARRAY_KEYS:
                if not merged.get(key) and isinstance(raw, list) and raw:
                    merged[key] = raw
                continue
            if key == "prescribed_medicines":
                if not merged.get(key) and isinstance(raw, list) and raw:
                    merged[key] = raw
                continue
            if _is_meaningful_string(raw) and not _is_meaningful_string(merged.get(key)):
                merged[key] = _str_val(raw)
    return merged


_DOCTOR_REG_IN_TEXT_RE = re.compile(
    r"(?:RMC|MCI|MMC|HPMC|HIMC|DMC|CN\.?\s*NO\.?|C\.?\s*N\.?\s*NO\.?|"
    r"council\s*(?:reg(?:istration)?|no\.?)|regn\.?\s*no\.?|reg\.?\s*no\.?|"
    r"regd\.?\s*no\.?|registration\s*no\.?)\s*[:\s#-]*"
    r"([A-Za-z0-9][A-Za-z0-9/\-]{1,20})",
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
PAYMENT_RECEIPT_PARAMS_SCHEMA = _param_object_schema(
    PAYMENT_RECEIPT_PARAM_KEYS, frozenset()
)

SYSTEM_PROMPT = """Indian medical claims OCR. Fill JSON; use "" / [] if absent. Signatures/stamps: "present" if visible but illegible.

EXTRACTION RULE (critical): Map EVERY visible value into the matching parameter fields.
Fill prescription_parameters AND invoice_parameters AND report_parameters AND payment_receipt_parameters
whenever those fields appear on the page — irrespective of document_category. Category is only a
best-guess label for downstream validation; do NOT leave fields empty just because they "belong to
another document type".
Examples: OPD/Rx page with footer GST → still fill invoice_parameters.gst_number.
Pharmacy bill with patient age → still fill invoice_parameters.patient_age (and prescription patient_* if present).
Lab report with doctor name → fill report + any overlapping patient_* fields.
UPI/GPay/PhonePe payment screenshot → fill payment_receipt_parameters (never invent values).

CONTENT TYPE (filled data only; ignore blank form lines). percents must sum to 100.
- handwritten: mostly hand-filled (cash memo) → hw%~100
- computer_generated: typed/printed → cg%~90-100; ignore footer signature alone
- Printed pharmacy/tax invoice/POS = computer_generated. Printed lab report = computer_generated.
- Payment app screenshots / bank confirmations = computer_generated.

CATEGORY (label only — still extract all visible fields into all buckets above)
- prescription | invoice | report | payment_receipt | other
- payment_receipt = UPI screenshots (GPay/PhonePe/Paytm/BHIM/Amazon Pay), bank transfer
  (NEFT/IMPS/RTGS/UTR), card payment confirmations, payment success/failure pages, SMS-style
  payment confirmations, readable cash receipts that are ONLY payment proof (not a medical bill).
- NEVER classify a payment screenshot as invoice. Medical bills/cash memos with line items = invoice.
- other = selfies/blank/unreadable/random non-claim images — NEVER pharmacy, clinic, hospital bills,
  or payment proofs
- is_medical_document=true for Rx, bills, lab reports, OPD receipts
- is_medical_document=false for pure payment_receipt / unrelated images
- Rx with no ₹ total → prescription (never invoice). Upside-down photos still medical.
- OPD SUMMARY / eye exam / refraction / glasses power sheets → prescription (eye_care), NOT report.
- report = lab/pathology/diagnostic test result sheets only (pathologist, specimen, analytes).

invoice_subtype: pharmacy | diagnostic | opd_consultation | dental | eye_care | uncertain | not_applicable
- pharmacy/chemist Bill of Supply/tax invoice/cash memo → pharmacy
- diagnostic centre bill → diagnostic; clinic "Received with thanks"/OPD fee → opd_consultation
- dental clinic → dental; optical/lens/frame → eye_care
- When document_category=payment_receipt → invoice_subtype=not_applicable, prescription_subtype=not_applicable

INVOICE: patient_*, invoice_number/date, provider_*, doctor_name, total_amount (digits only, primary total),
payment_mode, transaction_reference, authorized_stamp/signature.
MUST extract gst_number (15-char GSTIN only, no label) from top header/top-right if printed.
GSTIN may appear as **GSTIN:** **02AAHCHxxxxxZx** — strip * and return the code. Never FSSAI/CST/RST.
Pharmacy: drug_license_number (all DL lines, join "; "); medicine_details[] "name | Qty | Rate | Batch | Exp".
OPD: consultation_charges, registration_charges, service_details[].
Diagnostic: sample_collection_date (Registered On / Collected On / Received On), test_details[] "Test — Rs amt".
Dental/eye: item_details[] procedures or frame/lens lines.

PAYMENT_RECEIPT (mandatory when visible — do not invent):
- payment_mode: upi | bank_transfer | card | cash | wallet
- payment_amount: digits only (e.g. "684.00") — the paid amount, not balance
- transaction_date: ISO-like YYYY-MM-DD when possible, else as printed
- transaction_id: primary txn / UTR / UPI ref / bank ref
- payment_status: success | completed | failed | pending (normalize synonyms; "Payment Successful"→success)
Strongly recommended: payee_name, payer_name, upi_id (VPA like name@bank), bank_name,
reference_number (secondary ref), utr (duplicate transaction_id when UTR is the primary id).
Optional: payment_time, account_number_masked, ifsc, remarks.

prescription_subtype: opd | pharmacy | diagnostic | dental | eye_care | uncertain | not_applicable
Common: patient_*, consultation_date, clinic_hospital_*, doctor_name/qualification/registration_number,
doctor_signature/stamp, diagnosis, presenting_complaints, line_of_treatment, followup_*.
doctor_name: from letterhead or Prescribed by. Transliterate regional scripts to English when visible.
Use "" if the name is not printed or not readable — plain text only, no placeholders.
- opd/pharmacy: prescribed_medicines[{medicine,dosage}]; advised_tests if labs
- diagnostic: advised_tests[]; dental: tooth/treatment/procedure; eye: VA/power/glasses
eye_care / OPD SUMMARY / refraction sheets — extract EVERY visible clinical field:
- patient_name, patient_age, patient_gender from Age/Sex (e.g. "42 years 0 months /Female")
- consultation_date from Appt Dt / Note Dt; clinic_hospital_name from Facility; doctor_name from Doctor
- diagnosis: Systemic History / Visit reason when Chief Complaints is None/Nil
- presenting_complaints: Chief Complaints (use "" if literally None/Nil)
- visual_acuity_details: all VA + IOP lines
- eye_power_prescription: Distant/Near sphere summary for R/OD and L/OS
- glasses_contact_lens_prescription: full glasses prescription table text (powers + vision)
- treatment_advice: any advice / follow-up printed on the note
Also put footer GST No into invoice_parameters.gst_number when printed (even on Rx/OPD pages).
doctor_registration_number: read from rubber stamp / letterhead — labels include CN No, CN No.,
Council No, Reg No, Regd No, RMC/MCI/MMC/HPMC. Copy the number only (e.g. CN No: 12345 → "12345").
Do NOT use CR No / Patient Registration / Token No / Mobile / MR No as doctor_registration_number.
doctor_stamp / doctor_signature: "present" if visible but illegible; still try to read CN/Reg from stamp text.

REPORT: specific test_names (not section titles), test_results, dates, pathologist_*, laboratory_*.
sample_collection_date aliases (MUST fill when printed): Registered On, Sample Collected On,
Collected On, Collection Date, Received On, Sample Received, Drawn On, Scan Date, Examination Date.
Radiology/HRCT/CT/MRI/X-ray: no blood/urine sample — Registered On / Scan Date IS sample_collection_date.
report_date aliases: Reported On, Report Date, Date of Report, Released On.
Return dates as YYYY-MM-DD. Never leave sample_collection_date empty when Registered On is printed.
"""

FIELDS_SYSTEM_PROMPT = """Indian medical claims OCR. Fill JSON; use "" / [] if a value is not printed.
Put extracted values in parameters using the exact keys the caller requested.
Read EVERY attached page (multi-page PDFs). Headers AND footers matter:
Prescribed by / (Dr. Name) / Regn no are often at the bottom of a later page.
Map visible labels onto the requested keys. Do not invent values.
Dates as YYYY-MM-DD when possible. Signatures/stamps: "present" if visible but illegible.
Handwritten vs computer_generated percents must sum to 100.
doctor_name: transliterate regional-script letterhead names to English when visible; use "" if not found.
Text fields use "" when absent — never "present" (only doctor_signature / doctor_stamp use "present").
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
        "doctor_name": {"type": "string"},
        "doctor_registration_number": {"type": "string"},
        "doctor_stamp": {"type": "string"},
    },
    "required": ["doctor_name", "doctor_registration_number", "doctor_stamp"],
    "additionalProperties": False,
}

DOCTOR_REG_PROMPT = """Extract doctor_name and doctor_registration_number from prescription letterhead / stamp area.
doctor_name: full printed name; transliterate regional scripts to English. Use "" if not found.
doctor_registration_number: from stamp/signature area (CN No, Reg No, Regd No, RMC, MCI, MMC, HPMC) — number only.
Do NOT return CR No, Patient Registration, Token No, Room No, Mobile, Fee amounts, or barcodes.
doctor_stamp: "present" if a stamp/seal is visible (even if text is faint), else ""."""

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
            "enum": ["prescription", "invoice", "report", "payment_receipt", "other"],
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
        "payment_receipt_parameters": PAYMENT_RECEIPT_PARAMS_SCHEMA,
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
        "payment_receipt_parameters",
    ],
    "additionalProperties": False,
}

_VALID_CATEGORIES = frozenset(
    {"prescription", "invoice", "report", "payment_receipt", "other"}
)

_VALID_PAYMENT_MODES = frozenset(
    {"upi", "bank_transfer", "card", "cash", "wallet"}
)
_VALID_PAYMENT_STATUSES = frozenset(
    {"success", "completed", "failed", "pending"}
)

# CRM `name` field → document_category
_DOCUMENT_NAME_ALIASES = {
    "invoice": "invoice",
    "bill": "invoice",
    "tax_invoice": "invoice",
    "pharmacy_bill": "invoice",
    "prescription": "prescription",
    "rx": "prescription",
    "opd": "prescription",
    "opd_summary": "prescription",
    "report": "report",
    "lab": "report",
    "lab_report": "report",
    "diagnostic": "report",
    "payment_receipt": "payment_receipt",
    "payment": "payment_receipt",
    "payment_proof": "payment_receipt",
    "upi": "payment_receipt",
    "upi_payment": "payment_receipt",
    "bank_transfer": "payment_receipt",
}


def normalize_document_name_hint(raw: Any) -> str:
    """Map CRM document name hint to a valid document_category (or "")."""
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    if text in _VALID_CATEGORIES and text != "other":
        return text
    return _DOCUMENT_NAME_ALIASES.get(text, "")


_FIELD_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def normalize_extract_fields(raw: Any) -> Optional[Tuple[str, ...]]:
    """Dedupe caller field names from the API body. Any identifier is allowed."""
    if not raw or not isinstance(raw, (list, tuple)):
        return None
    normalized: List[str] = []
    for item in raw:
        key = str(item or "").strip()
        if _FIELD_NAME_RE.match(key) and key not in normalized:
            normalized.append(key)
    return tuple(normalized) if normalized else None


def _schema_for_field(key: str) -> Dict[str, Any]:
    if key == "prescribed_medicines":
        return {"type": "array", "items": _MEDICINE_ITEM_SCHEMA}
    if key in _ALL_CLAIM_ARRAY_KEYS:
        return _STRING_ARRAY
    return {"type": "string"}


def _parameters_schema_from_fields(fields: Sequence[str]) -> Dict[str, Any]:
    """JSON schema for parameters — keys come from the API `fields` list."""
    return {
        "type": "object",
        "properties": {key: _schema_for_field(key) for key in fields},
        "required": list(fields),
        "additionalProperties": False,
    }


def _build_slim_document_schema(fields: Sequence[str]) -> Dict[str, Any]:
    """OpenAI schema driven by the request `fields` list, not a hardcoded catalog."""
    return {
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
            "parameters": _parameters_schema_from_fields(fields),
        },
        "required": [
            "document_type",
            "content_handwritten_percent",
            "content_computer_generated_percent",
            "is_medical_document",
            "non_medical_reason",
            "parameters",
        ],
        "additionalProperties": False,
    }


def use_textract_for_category_hint(hint: str) -> bool:
    """When CRM sends name, skip Textract except for invoices (saves ~15–30s per file)."""
    if not textract_enabled():
        return False
    normalized = normalize_document_name_hint(hint)
    if not normalized:
        return True
    only_invoice = (os.getenv("TEXTRACT_HINT_INVOICE_ONLY") or "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if only_invoice:
        return normalized == "invoice"
    return True


def _openai_max_tokens_for_hint(hint: str) -> int:
    """Smaller cap when category is known — faster responses, same field coverage."""
    if normalize_document_name_hint(hint):
        try:
            return max(800, int(os.getenv("OPENAI_HINT_MAX_TOKENS", "1400")))
        except ValueError:
            return 1400
    try:
        return max(1200, int(os.getenv("OPENAI_MAX_TOKENS", "2200")))
    except ValueError:
        return 2200


def _openai_max_tokens_for_request(
    hint: str,
    extract_fields: Optional[Sequence[str]] = None,
) -> int:
    """Tighter token cap when CRM sends an explicit field list."""
    base = _openai_max_tokens_for_hint(hint)
    if not extract_fields:
        return base
    estimated = min(1800, max(700, 500 + len(extract_fields) * 80))
    try:
        cap = int(os.getenv("OPENAI_FIELDS_MAX_TOKENS", str(estimated)))
    except ValueError:
        cap = estimated
    return max(400, min(cap, base))


def _fields_need_doctor_reg_refine(extract_fields: Optional[Sequence[str]]) -> bool:
    if not extract_fields:
        return True
    return any(
        key in extract_fields
        for key in ("doctor_registration_number", "doctor_stamp", "doctor_name")
    )


def _fields_need_gst_refine(extract_fields: Optional[Sequence[str]]) -> bool:
    if not extract_fields:
        return True
    return "gst_number" in extract_fields or "drug_license_number" in extract_fields


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


def _is_usable_doctor_name(raw: Any) -> bool:
    """True only for a real printed doctor name — placeholders become ""."""
    name = _str_val(raw)
    if not name:
        return False
    low = name.lower()
    if low == "present" or low in _PLACEHOLDER_VALUES:
        return False
    if "..." in name or "…" in name:
        return False
    if re.fullmatch(r"dr\.?", name, flags=re.I):
        return False
    core = re.sub(r"^dr\.?\s*", "", name, flags=re.I).strip(" .")
    if len(core) < 3 or not re.search(r"[A-Za-z\u0900-\u0dff]", core):
        return False
    return True


def _normalize_doctor_name_value(params: Dict[str, Any]) -> None:
    """Strip Prescribed by / parentheses wrappers from a printed doctor name."""
    if "doctor_name" not in params:
        return
    name = _str_val(params.get("doctor_name"))
    if not _is_usable_doctor_name(name):
        params["doctor_name"] = ""
        return
    name = re.sub(r"^(?:prescribed\s*by\s*:?\s*)", "", name, flags=re.I)
    name = name.strip("() ").strip()
    name = re.sub(r"\s+", " ", name)
    params["doctor_name"] = name


def _normalize_doctor_registration(params: Dict[str, Any]) -> None:
    """Fill doctor_registration_number from stamp/name text; support CN No. labels."""
    _normalize_doctor_name_value(params)
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


_UNICODE_SPACE_RE = re.compile(r"[\u00a0\u202f\u2007\u2009\u200b]")
_TIME_SUFFIX_RE = re.compile(
    r"\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?$",
)
_MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_NAMED_DATE_RE = re.compile(
    r"^(\d{1,2})[/\-.\s]([A-Za-z]{3,9})[/\-.\s](\d{2,4})$"
)
_YMD_DATE_RE = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$")
_DMY_DATE_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")
_DATE_VALUE_RE = (
    r"(\d{1,2}[/\-.\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"[/\-.\s]\d{2,4}"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?)?"
    r"|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?)?)"
)
_SAMPLE_COLLECTION_LABEL_RE = re.compile(
    r"(?:"
    r"sample\s*collected(?:\s*on)?"
    r"|sample\s*collection\s*date"
    r"|sample\s*received(?:\s*on)?"
    r"|sample\s*date"
    r"|collected\s*on"
    r"|collection\s*date"
    r"|registered\s*on"
    r"|registration\s*date"
    r"|date\s*of\s*registration"
    r"|received\s*on"
    r"|received\s*date"
    r"|drawn\s*on"
    r"|specimen\s*(?:collected|date)"
    r"|scan\s*date"
    r"|date\s*of\s*scan"
    r"|examination\s*date"
    r"|date\s*of\s*examination"
    r"|date\s*of\s*(?:investigation|procedure|study)"
    r"|investigation\s*date"
    r"|procedure\s*date"
    r")\s*[:\-]?\s*"
    + _DATE_VALUE_RE,
    re.IGNORECASE,
)
_REPORT_DATE_LABEL_RE = re.compile(
    r"(?:"
    r"reported\s*on"
    r"|report\s*date"
    r"|date\s*of\s*report"
    r"|date\s*reported"
    r"|released\s*on"
    r"|report\s*released"
    r"|verified\s*on"
    r")\s*[:\-]?\s*"
    + _DATE_VALUE_RE,
    re.IGNORECASE,
)


def _collapse_ws(raw: str) -> str:
    return re.sub(r"\s+", " ", _UNICODE_SPACE_RE.sub(" ", raw)).strip()


def _normalize_to_iso_date(raw: Any) -> str:
    """Normalize printed Indian dates to YYYY-MM-DD; keep original if unparseable."""
    original = _str_val(raw)
    text = _TIME_SUFFIX_RE.sub("", _collapse_ws(original)).strip(" .,;")
    if not text:
        return ""

    named = _NAMED_DATE_RE.match(text)
    if named:
        day_s, month_s, year_s = named.groups()
        month = _MONTH_LOOKUP.get(month_s.lower()) or _MONTH_LOOKUP.get(month_s.lower()[:3])
        if month:
            year = int(year_s)
            if year < 100:
                year += 2000
            day = int(day_s)
            if 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"

    ymd = _YMD_DATE_RE.match(text)
    if ymd:
        year, month, day = (int(part) for part in ymd.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

    dmy = _DMY_DATE_RE.match(text)
    if dmy:
        day, month, year = (int(part) for part in dmy.groups())
        if year < 100:
            year += 2000
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return original


def _extract_labeled_date(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(_UNICODE_SPACE_RE.sub(" ", text))
    if not match:
        return ""
    return _normalize_to_iso_date(match.group(1))


def _fill_labeled_dates_from_text(data: Dict[str, Any], text: str) -> None:
    """Fill empty sample_collection_date / report_date from labeled PDF/OCR text."""
    if not text or not text.strip():
        return
    sample = _extract_labeled_date(text, _SAMPLE_COLLECTION_LABEL_RE)
    report = _extract_labeled_date(text, _REPORT_DATE_LABEL_RE)
    if not sample and not report:
        return

    assignments = (
        ("parameters", "sample_collection_date", sample),
        ("report_parameters", "sample_collection_date", sample),
        ("invoice_parameters", "sample_collection_date", sample),
        ("parameters", "report_date", report),
        ("report_parameters", "report_date", report),
    )
    for bucket_key, field, value in assignments:
        if not value:
            continue
        bucket = data.get(bucket_key)
        if not isinstance(bucket, dict):
            bucket = {}
            data[bucket_key] = bucket
        if not _is_meaningful_string(bucket.get(field)):
            bucket[field] = value


def _normalize_report_fields(params: Dict[str, Any]) -> None:
    for key in ("sample_collection_date", "report_date"):
        val = _str_val(params.get(key))
        if val:
            params[key] = _normalize_to_iso_date(val)


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
    sample = _str_val(params.get("sample_collection_date"))
    if sample:
        params["sample_collection_date"] = _normalize_to_iso_date(sample)
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


_PAYMENT_STATUS_ALIASES = {
    "successful": "success",
    "successfully": "success",
    "paid": "success",
    "done": "success",
    "complete": "completed",
    "payment successful": "success",
    "payment completed": "completed",
    "transaction successful": "success",
    "transaction failed": "failed",
    "payment failed": "failed",
    "failure": "failed",
    "declined": "failed",
    "processing": "pending",
    "in progress": "pending",
    "awaiting": "pending",
}

_PAYMENT_MODE_ALIASES = {
    "gpay": "upi",
    "google pay": "upi",
    "phonepe": "upi",
    "paytm upi": "upi",
    "bhim": "upi",
    "amazon pay": "upi",
    "amazon pay upi": "upi",
    "upi payment": "upi",
    "neft": "bank_transfer",
    "imps": "bank_transfer",
    "rtgs": "bank_transfer",
    "rtps": "bank_transfer",
    "bank transfer": "bank_transfer",
    "net banking": "bank_transfer",
    "netbanking": "bank_transfer",
    "credit card": "card",
    "debit card": "card",
    "credit": "card",
    "debit": "card",
    "visa": "card",
    "mastercard": "card",
    "rupay": "card",
    "paytm wallet": "wallet",
    "wallet": "wallet",
}


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


def _is_plausible_upi_id(value: Any) -> bool:
    """Accept real UPI VPAs; reject emails / letterhead addresses (info@hospital)."""
    raw = _str_val(value)
    if "@" not in raw or " " in raw:
        return False
    local, _, handle = raw.partition("@")
    if not local or not handle:
        return False
    handle_l = handle.lower()
    local_l = local.lower()
    if "." in handle_l:
        return False
    if local_l in {"info", "admin", "support", "contact", "hello", "mail", "email"}:
        return False
    if handle_l in _KNOWN_UPI_HANDLES:
        return True
    if re.match(r"^(ok|pt|yb|ibl|axl|apl|wa)", handle_l):
        return True
    if len(handle_l) <= 12 and any(ch.isdigit() for ch in local):
        return True
    return False


def _normalize_payment_amount(raw: Any) -> str:
    text = _str_val(raw)
    if not text:
        return ""
    clean = re.sub(r"[^\d.]", "", text.replace(",", ""))
    if not clean:
        return ""
    if clean.count(".") > 1:
        parts = clean.split(".")
        clean = "".join(parts[:-1]) + "." + parts[-1]
    try:
        if float(clean) <= 0:
            return ""
    except ValueError:
        return ""
    return clean


def _normalize_payment_status(raw: Any) -> str:
    text = _str_val(raw).lower()
    if not text:
        return ""
    if text in _VALID_PAYMENT_STATUSES:
        return text
    aliased = _PAYMENT_STATUS_ALIASES.get(text)
    if aliased:
        return aliased
    for key, status in _PAYMENT_STATUS_ALIASES.items():
        if key in text:
            return status
    for status in _VALID_PAYMENT_STATUSES:
        if status in text:
            return status
    return text


def _normalize_payment_mode(raw: Any) -> str:
    text = _str_val(raw).lower()
    if not text:
        return ""
    if text in _VALID_PAYMENT_MODES:
        return text
    aliased = _PAYMENT_MODE_ALIASES.get(text)
    if aliased:
        return aliased
    for key, mode in sorted(_PAYMENT_MODE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if key in text:
            return mode
    for mode in _VALID_PAYMENT_MODES:
        if mode in text:
            return mode
    return text


def _normalize_bank_name(raw: Any) -> str:
    text = _str_val(raw)
    if not text:
        return ""
    # Bare "Axis" is usually an auto-refraction column, not Axis Bank.
    low = text.lower().strip()
    if low in {"axis", "yes", "union", "federal"}:
        return ""
    return text


def _normalize_payment_receipt_fields(params: Dict[str, Any]) -> None:
    """Normalize payment proof fields; clear placeholders; sync utr/transaction_id."""
    for key in PAYMENT_RECEIPT_PARAM_KEYS:
        val = _str_val(params.get(key))
        if val.lower() in _PLACEHOLDER_VALUES:
            params[key] = ""
        else:
            params[key] = val

    params["payment_amount"] = _normalize_payment_amount(params.get("payment_amount"))
    params["payment_status"] = _normalize_payment_status(params.get("payment_status"))
    params["payment_mode"] = _normalize_payment_mode(params.get("payment_mode"))
    params["bank_name"] = _normalize_bank_name(params.get("bank_name"))

    upi = _str_val(params.get("upi_id"))
    params["upi_id"] = upi if _is_plausible_upi_id(upi) else ""

    payee = _str_val(params.get("payee_name"))
    if len(payee) < 3 or payee.lower() in {"none", "nil", "n/a", "ry none", "tal"}:
        params["payee_name"] = ""
    elif re.match(r"^(ry|tal|history)\b", payee, re.I):
        params["payee_name"] = ""

    txn = _str_val(params.get("transaction_id"))
    utr = _str_val(params.get("utr"))
    ref = _str_val(params.get("reference_number"))
    if not txn and utr:
        params["transaction_id"] = utr
        txn = utr
    if not utr and txn and re.search(r"utr|^\d{12,}$", txn, re.IGNORECASE):
        params["utr"] = txn
        utr = txn
    if not ref and txn:
        params["reference_number"] = txn


def _payment_amount_present(pay: Dict[str, Any]) -> bool:
    amount = _normalize_payment_amount(pay.get("payment_amount"))
    return bool(amount)


def _payment_txn_id_present(pay: Dict[str, Any]) -> bool:
    return (
        _is_filled(pay, "transaction_id")
        or _is_filled(pay, "reference_number")
        or _is_filled(pay, "utr")
    )


def _payment_receipt_extra_checks(
    pay: Dict[str, Any],
) -> List[Tuple[str, bool]]:
    return [("transaction_id_or_reference_or_utr", _payment_txn_id_present(pay))]


def _looks_like_payment_receipt(pay: Dict[str, Any]) -> bool:
    """Strong payment-proof signals only — ignore email/Axis/IOP false positives."""
    if not isinstance(pay, dict):
        return False
    amount = _payment_amount_present(pay)
    status = _is_filled(pay, "payment_status")
    txn = _payment_txn_id_present(pay)
    mode = _is_filled(pay, "payment_mode")
    upi = _is_plausible_upi_id(pay.get("upi_id"))

    # Need a real payment outcome or transfer rail, plus amount or txn id.
    if status and (amount or txn) and (mode or upi or txn):
        return True
    if amount and txn and (status or mode):
        return True
    if upi and amount and (status or txn or mode):
        return True
    if status and txn and mode:
        return True
    return False


def _payment_extraction_score(pay: Dict[str, Any]) -> int:
    score = 0
    weighted = (
        ("payment_amount", 2),
        ("transaction_date", 1),
        ("transaction_id", 2),
        ("utr", 2),
        ("reference_number", 1),
        ("payment_status", 2),
        ("payment_mode", 1),
        ("upi_id", 2),
        ("payee_name", 1),
        ("payer_name", 1),
        ("bank_name", 1),
    )
    for key, weight in weighted:
        if key == "payment_amount":
            if _payment_amount_present(pay):
                score += weight
        elif key == "upi_id":
            if _is_plausible_upi_id(pay.get("upi_id")):
                score += weight
        elif key == "bank_name":
            if _normalize_bank_name(pay.get("bank_name")):
                score += weight
        elif _is_filled(pay, key):
            score += weight
    return score


def _non_medical_reason_suggests_payment(data: Dict[str, Any]) -> bool:
    reason = _str_val(data.get("non_medical_reason")).lower()
    if not reason:
        return False
    return any(
        hint in reason
        for hint in (
            "payment",
            "upi",
            "gpay",
            "phonepe",
            "paytm",
            "neft",
            "imps",
            "utr",
            "transaction",
            "wallet",
            "bank transfer",
        )
    )


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
    # Eye OPD: reuse power summary into glasses field when table text missing.
    if _is_filled(params, "eye_power_prescription") and not _is_filled(
        params, "glasses_contact_lens_prescription"
    ):
        params["glasses_contact_lens_prescription"] = _str_val(
            params.get("eye_power_prescription")
        )
    # Age/Sex often returns "42 years 0 months" — keep digits at minimum readable.
    age = _str_val(params.get("patient_age"))
    if age:
        m = re.search(r"(\d{1,3})", age)
        if m and not re.search(r"\d", age[m.end() :][:3]):
            # keep full text if present; only normalize bare age
            pass
        params["patient_age"] = age
    gender = _str_val(params.get("patient_gender")).lower()
    if gender in ("female", "f"):
        params["patient_gender"] = "F"
    elif gender in ("male", "m"):
        params["patient_gender"] = "M"


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
    prescription_params: Dict[str, Any] | None = None,
) -> str:
    if category != "invoice":
        return category
    if not _is_filled(invoice_params, "total_amount"):
        rx = prescription_params or {}
        if _count_extracted_fields(rx) >= 3:
            return "prescription"
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


def _looks_like_invoice_bill(inv: Dict[str, Any]) -> bool:
    """True bill/cash-memo signals — not GSTIN/letterhead alone (common on OPD summaries)."""
    return bool(
        _is_filled(inv, "total_amount")
        or _is_filled(inv, "medicine_details")
        or _is_filled(inv, "test_details")
        or _is_filled(inv, "item_details")
        or (
            _is_filled(inv, "invoice_number")
            and _is_filled(inv, "invoice_date")
            and (_is_filled(inv, "provider_name") or _is_filled(inv, "patient_name"))
        )
    )


def _has_real_invoice_structure(inv: Dict[str, Any]) -> bool:
    """Medical/pharmacy bill structure — not a UPI screenshot with only an amount."""
    return bool(
        _is_filled(inv, "medicine_details")
        or _is_filled(inv, "test_details")
        or _is_filled(inv, "item_details")
        or _is_filled(inv, "service_details")
        or (
            _is_filled(inv, "invoice_number")
            and (
                _is_filled(inv, "provider_name")
                or _is_filled(inv, "gst_number")
                or _is_filled(inv, "drug_license_number")
            )
        )
        or (
            _is_filled(inv, "provider_name")
            and _is_filled(inv, "total_amount")
            and (
                _is_filled(inv, "gst_number")
                or _is_filled(inv, "invoice_number")
                or _is_filled(inv, "consultation_charges")
            )
        )
    )


def _seed_payment_from_invoice_leak(
    pay: Dict[str, Any], inv: Dict[str, Any]
) -> None:
    """Copy UPI amount / txn ref leaked into invoice_* into payment_receipt_*."""
    if not _payment_amount_present(pay) and _is_filled(inv, "total_amount"):
        pay["payment_amount"] = _normalize_payment_amount(inv.get("total_amount"))
    if not _is_filled(pay, "payment_mode") and _is_filled(inv, "payment_mode"):
        pay["payment_mode"] = _normalize_payment_mode(inv.get("payment_mode"))
    if not _payment_txn_id_present(pay) and _is_filled(inv, "transaction_reference"):
        ref = _str_val(inv.get("transaction_reference"))
        pay["transaction_id"] = ref
        if not _str_val(pay.get("reference_number")):
            pay["reference_number"] = ref
    if not _is_filled(pay, "transaction_date") and _is_filled(inv, "invoice_date"):
        pay["transaction_date"] = _str_val(inv.get("invoice_date"))
    if not _is_filled(pay, "payer_name") and _is_filled(inv, "patient_name"):
        # Only when other payment rails already present (avoid Rx patient → payer).
        if (
            _payment_txn_id_present(pay)
            or _is_plausible_upi_id(pay.get("upi_id"))
            or _is_filled(pay, "payment_status")
        ):
            pay["payer_name"] = _str_val(inv.get("patient_name"))
    _normalize_payment_receipt_fields(pay)


def _is_payment_proof_not_medical_bill(
    pay: Dict[str, Any], inv: Dict[str, Any]
) -> bool:
    """UPI/bank payment screenshot mistaken for invoice (amount → total_amount)."""
    if not _looks_like_payment_receipt(pay):
        return False
    if _has_real_invoice_structure(inv):
        return False
    return True


def _apply_payment_receipt_category(data: Dict[str, Any], pay: Dict[str, Any]) -> None:
    data["is_medical_document"] = False
    data["document_category"] = "payment_receipt"
    data["invoice_subtype"] = "not_applicable"
    data["prescription_subtype"] = "not_applicable"
    data["non_medical_reason"] = ""
    data["payment_receipt_parameters"] = pay


def _recover_medical_classification(data: Dict[str, Any]) -> None:
    """Undo false 'other' when extraction or reason indicates a real bill/Rx/report/payment."""
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
    pay = _normalize_params(
        data.get("payment_receipt_parameters"), PAYMENT_RECEIPT_PARAM_KEYS, frozenset()
    )
    _normalize_payment_receipt_fields(pay)
    _seed_payment_from_invoice_leak(pay, inv)

    inv_score = _invoice_extraction_score(inv)
    rx_score = _count_extracted_fields(rx)
    rep_score = _count_extracted_fields(rep)
    pay_score = _payment_extraction_score(pay)
    original_category = str(data.get("document_category", "")).strip().lower()
    bill_like = _looks_like_invoice_bill(inv)
    payment_like = _looks_like_payment_receipt(pay)
    payment_not_bill = _is_payment_proof_not_medical_bill(pay, inv)

    # PhonePe / UPI / bank screenshots: never keep as invoice when payment proof is strong
    # and there is no real bill structure (invoice no / line items / provider+GST).
    if payment_not_bill and rx_score < 3 and rep_score < 3:
        _apply_payment_receipt_category(data, pay)
        return

    # Prefer filled medical docs over weak / false payment cues (email≠UPI, Axis≠bank).
    if (
        rx_score >= 3
        and (
            original_category == "prescription"
            or not bill_like
            or rx_score >= inv_score + 2
        )
        and (not payment_like or rx_score >= pay_score)
    ):
        data["is_medical_document"] = True
        data["document_category"] = "prescription"
        data["invoice_subtype"] = "not_applicable"
        data["prescription_subtype"] = _infer_prescription_subtype(
            rx, str(data.get("prescription_subtype", "uncertain"))
        )
        data["non_medical_reason"] = ""
        data["payment_receipt_parameters"] = {
            key: "" for key in PAYMENT_RECEIPT_PARAM_KEYS
        }
        return

    if (
        bill_like
        and not payment_not_bill
        and (inv_score >= 3 or (inv_score >= 2 and _is_filled(inv, "medicine_details")))
    ):
        data["is_medical_document"] = True
        data["document_category"] = "invoice"
        data["non_medical_reason"] = ""
        if _is_filled(inv, "medicine_details") or _is_filled(inv, "drug_license_number"):
            data["invoice_subtype"] = "pharmacy"
        elif _looks_like_opd_consultation_invoice(inv):
            data["invoice_subtype"] = "opd_consultation"
        if not payment_like:
            data["payment_receipt_parameters"] = {
                key: "" for key in PAYMENT_RECEIPT_PARAM_KEYS
            }
        return

    if (
        bill_like
        and not payment_not_bill
        and inv_score >= 2
        and _looks_like_opd_consultation_invoice(inv)
    ):
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
        data["payment_receipt_parameters"] = {
            key: "" for key in PAYMENT_RECEIPT_PARAM_KEYS
        }
        return

    if rep_score >= 3:
        data["is_medical_document"] = True
        data["document_category"] = "report"
        data["invoice_subtype"] = "not_applicable"
        data["non_medical_reason"] = ""
        return

    # Payment screenshots only after medical buckets are weak.
    if payment_like and rx_score < 3 and rep_score < 3 and (
        original_category in ("payment_receipt", "other", "", "invoice")
        or pay_score >= inv_score + 2
        or (not bill_like and pay_score >= 4)
        or payment_not_bill
    ):
        _apply_payment_receipt_category(data, pay)
        return

    if (
        original_category == "payment_receipt"
        and payment_like
        and pay_score >= 4
        and rx_score < 3
        and rep_score < 3
    ):
        _apply_payment_receipt_category(data, pay)
        return

    if payment_like or (pay_score >= 4 and _non_medical_reason_suggests_payment(data)):
        if rx_score < 3 and rep_score < 3:
            _apply_payment_receipt_category(data, pay)
            return

    if _non_medical_reason_suggests_bill(data) and not payment_not_bill:
        data["is_medical_document"] = True
        data["document_category"] = "invoice"
        reason = _str_val(data.get("non_medical_reason")).lower()
        if any(h in reason for h in ("consultation", "clinic", "opd", "doctor", "physician")):
            data["invoice_subtype"] = "opd_consultation"
        else:
            data["invoice_subtype"] = "pharmacy"
        data["non_medical_reason"] = ""
        return

    if _non_medical_reason_suggests_payment(data) and pay_score >= 4 and rx_score < 3:
        _apply_payment_receipt_category(data, pay)

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
    pay = _normalize_params(
        data.get("payment_receipt_parameters"), PAYMENT_RECEIPT_PARAM_KEYS, frozenset()
    )
    return (
        _count_extracted_fields(rx) == 0
        and _count_extracted_fields(inv) == 0
        and _count_extracted_fields(rep) == 0
        and _count_extracted_fields(pay) == 0
    )


def _resolve_non_medical(
    data: Dict[str, Any], category: str
) -> Tuple[str, str, bool]:
    """Return (category, reason, is_claim_document).

    is_claim_document covers medical docs and payment proofs.
    """
    reason = _str_val(data.get("non_medical_reason"))
    is_medical = bool(data.get("is_medical_document", True))

    if category == "payment_receipt":
        return "payment_receipt", "", True

    if category == "other":
        return "other", reason or "Not a medical claim document", False

    if not is_medical and not _non_medical_reason_suggests_bill(data):
        return (
            "other",
            reason or "Image is not a prescription, invoice, lab report, or payment proof",
            False,
        )

    if _all_medical_blocks_empty(data) and not _non_medical_reason_suggests_bill(data):
        return (
            "other",
            reason or "No prescription, bill, report, or payment content found in the image",
            False,
        )

    return category, "", True


def _apply_category_hint(data: Dict[str, Any], category_hint: Optional[str]) -> None:
    """Lock document_category to CRM-provided name when valid."""
    hint = normalize_document_name_hint(category_hint)
    if not hint:
        return
    data["document_category"] = hint
    data["non_medical_reason"] = ""
    if hint == "payment_receipt":
        data["is_medical_document"] = False
        data["invoice_subtype"] = "not_applicable"
        data["prescription_subtype"] = "not_applicable"
    elif hint == "prescription":
        data["is_medical_document"] = True
        data["invoice_subtype"] = "not_applicable"
        # Drop false payment cues (email/Axis) on Rx pages.
        data["payment_receipt_parameters"] = {
            key: "" for key in PAYMENT_RECEIPT_PARAM_KEYS
        }
    elif hint == "invoice":
        data["is_medical_document"] = True
        data["prescription_subtype"] = "not_applicable"
    elif hint == "report":
        data["is_medical_document"] = True
        data["invoice_subtype"] = "not_applicable"
        data["prescription_subtype"] = "not_applicable"
        data["payment_receipt_parameters"] = {
            key: "" for key in PAYMENT_RECEIPT_PARAM_KEYS
        }


def _with_request_name(
    result: Dict[str, Any],
    hint: str,
    extract_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Echo CRM request name/fields on each result for downstream validation matching."""
    if hint:
        result["name"] = hint
    if extract_fields:
        result["fields"] = list(extract_fields)
    return result


def _build_public_response(
    url: str,
    data: Dict[str, Any],
    category_hint: Optional[str] = None,
    extract_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    hint = normalize_document_name_hint(category_hint)
    if hint:
        # CRM label wins — skip auto recovery that flips invoice ↔ payment ↔ Rx.
        _apply_category_hint(data, hint)
    else:
        _recover_medical_classification(data)

    if extract_fields:
        category = hint or str(data.get("document_category", "other"))
        if category not in _VALID_CATEGORIES:
            category = "other"
        doc_type = _apply_document_type(data)
        content_hw, content_cg = _content_percent_split(data)
        parameters = _parameters_from_extraction(data, extract_fields)
        completeness, missing_parameters = _completeness(
            parameters, extract_fields, ()
        )
        invoice_subtype = str(data.get("invoice_subtype") or "not_applicable")
        if invoice_subtype not in _VALID_INVOICE_SUBTYPES:
            invoice_subtype = "not_applicable"
        prescription_subtype = str(data.get("prescription_subtype") or "not_applicable")
        if prescription_subtype not in _VALID_PRESCRIPTION_SUBTYPES:
            prescription_subtype = "not_applicable"
        if category == "report":
            invoice_subtype = "not_applicable"
            prescription_subtype = "not_applicable"
        elif category == "prescription":
            invoice_subtype = "not_applicable"
            if prescription_subtype in ("not_applicable", "uncertain", ""):
                prescription_subtype = _infer_prescription_subtype(
                    parameters, prescription_subtype or "uncertain"
                )
        elif category == "invoice":
            prescription_subtype = "not_applicable"
        elif category == "payment_receipt":
            invoice_subtype = "not_applicable"
            prescription_subtype = "not_applicable"
        return _with_request_name(
            {
                "url": url,
                "is_medical_document": category != "payment_receipt" and category != "other",
                "document_type": doc_type,
                "document_category": category,
                "invoice_subtype": invoice_subtype,
                "prescription_subtype": prescription_subtype,
                "handwritten_percent": content_hw,
                "computer_generated_percent": content_cg,
                "completeness_percent": completeness,
                "parameters": parameters,
                "missing_parameters": missing_parameters,
                "message": "",
            },
            hint,
            extract_fields,
        )

    category = str(data.get("document_category", "other"))
    if category not in _VALID_CATEGORIES:
        category = "other"

    # Safety net only when CRM did not supply a category hint.
    if not hint and category == "invoice":
        inv_probe = _normalize_params(
            data.get("invoice_parameters"), INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS
        )
        pay_probe = _normalize_params(
            data.get("payment_receipt_parameters"),
            PAYMENT_RECEIPT_PARAM_KEYS,
            frozenset(),
        )
        _normalize_payment_receipt_fields(pay_probe)
        _seed_payment_from_invoice_leak(pay_probe, inv_probe)
        if _is_payment_proof_not_medical_bill(pay_probe, inv_probe):
            _apply_payment_receipt_category(data, pay_probe)
            category = "payment_receipt"

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
    rx_for_category = _normalize_params(
        data.get("prescription_parameters"),
        PRESCRIPTION_PARAM_KEYS,
        _PRESCRIPTION_ARRAY_KEYS,
    )
    if category != "payment_receipt" and not hint:
        category = _correct_document_category(
            doc_type, category, inv_params, rx_for_category
        )
    if hint:
        category = hint
        data["document_category"] = hint
        non_medical_reason, is_claim_doc = "", True
    else:
        category, non_medical_reason, is_claim_doc = _resolve_non_medical(
            data, category
        )

    if not is_claim_doc:
        return _with_request_name(
            {
                "url": url,
                "is_medical_document": False,
                "document_type": doc_type,
                "document_category": "other",
                "invoice_subtype": "not_applicable",
                "prescription_subtype": "not_applicable",
                "handwritten_percent": content_hw,
                "computer_generated_percent": content_cg,
                "completeness_percent": 0.0,
                "parameters": _finalize_parameters({}, extract_fields),
                "missing_parameters": ["not_a_medical_document"],
                "message": non_medical_reason,
            },
            hint,
            extract_fields,
        )

    prescription_subtype = "not_applicable"
    invoice_subtype = "not_applicable"

    # Always normalize every bucket, then merge into one parameters map for main-service.
    rx_params = _normalize_params(
        data.get("prescription_parameters"),
        PRESCRIPTION_PARAM_KEYS,
        _PRESCRIPTION_ARRAY_KEYS,
    )
    _normalize_prescription_fields(rx_params)
    _normalize_doctor_registration(rx_params)
    data["prescription_parameters"] = rx_params

    inv_params = _normalize_params(
        data.get("invoice_parameters"), INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS
    )
    _normalize_invoice_fields(inv_params)
    data["invoice_parameters"] = inv_params

    rep_params = _normalize_params(
        data.get("report_parameters"), REPORT_PARAM_KEYS, _REPORT_ARRAY_KEYS
    )
    _normalize_report_fields(rep_params)
    data["report_parameters"] = rep_params

    pay_params = _normalize_params(
        data.get("payment_receipt_parameters"), PAYMENT_RECEIPT_PARAM_KEYS, frozenset()
    )
    _normalize_payment_receipt_fields(pay_params)
    data["payment_receipt_parameters"] = pay_params

    if category == "payment_receipt":
        parameters = pay_params
        extra_checks = _payment_receipt_extra_checks(parameters)
        required_fields = (
            tuple(extract_fields)
            if extract_fields
            else PAYMENT_RECEIPT_REQUIRED
        )
        unified = _merge_claim_field_buckets(
            parameters, rx_params, inv_params, rep_params
        )
        check_params = unified if extract_fields else parameters
        completeness, missing_parameters = _completeness(
            check_params,
            required_fields,
            tuple(extra_checks) if not extract_fields else (),
        )
        return _with_request_name(
            {
                "url": url,
                "is_medical_document": False,
                "document_type": doc_type if doc_type != "uncertain" else "computer_generated",
                "document_category": "payment_receipt",
                "invoice_subtype": "not_applicable",
                "prescription_subtype": "not_applicable",
                "handwritten_percent": content_hw,
                "computer_generated_percent": content_cg,
                "completeness_percent": completeness,
                "parameters": _finalize_parameters(unified, extract_fields),
                "missing_parameters": missing_parameters,
                "message": "",
            },
            hint,
            extract_fields,
        )

    if category == "prescription":
        parameters = rx_params
        prescription_subtype = _infer_prescription_subtype(
            parameters, str(data.get("prescription_subtype", "uncertain"))
        )
        required = (
            tuple(extract_fields)
            if extract_fields
            else PRESCRIPTION_SUBTYPE_REQUIRED.get(
                prescription_subtype, PRESCRIPTION_REQUIRED
            )
        )
        extra_checks = (
            ()
            if extract_fields
            else _prescription_subtype_extra_checks(prescription_subtype, parameters)
        )
        completeness, missing_parameters = _completeness(
            parameters,
            required,
            tuple(extra_checks),
        )
    elif category == "invoice":
        parameters = inv_params
        invoice_subtype = _infer_invoice_subtype(
            parameters, str(data.get("invoice_subtype", "uncertain"))
        )
        extra_checks = (
            ()
            if extract_fields
            else _invoice_subtype_extra_checks(invoice_subtype, parameters)
        )
        required = tuple(extract_fields) if extract_fields else INVOICE_REQUIRED
        completeness, missing_parameters = _completeness(
            parameters,
            required,
            tuple(extra_checks),
        )
    else:
        parameters = rep_params
        _fix_report_content_classification(data, parameters)
        parameters = _normalize_params(
            data.get("report_parameters"), REPORT_PARAM_KEYS, _REPORT_ARRAY_KEYS
        )
        doc_type = _apply_document_type(data)
        content_hw, content_cg = _content_percent_split(data)
        required = tuple(extract_fields) if extract_fields else REPORT_REQUIRED
        completeness, missing_parameters = _completeness(parameters, required, ())

    # Surface every filled field from all buckets — classification does not hide values.
    unified = _merge_claim_field_buckets(
        rx_params, inv_params, rep_params, pay_params, parameters
    )

    return _with_request_name(
        {
            "url": url,
            "is_medical_document": True,
            "document_type": doc_type,
            "document_category": category,
            "invoice_subtype": invoice_subtype,
            "prescription_subtype": prescription_subtype,
            "handwritten_percent": content_hw,
            "computer_generated_percent": content_cg,
            "completeness_percent": completeness,
            "parameters": _finalize_parameters(unified, extract_fields),
            "missing_parameters": missing_parameters,
            "message": "",
        },
        hint,
        extract_fields,
    )








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
    buckets = []
    rx_raw = data.get("prescription_parameters")
    if isinstance(rx_raw, dict):
        buckets.append(rx_raw)
    params_raw = data.get("parameters")
    if isinstance(params_raw, dict):
        buckets.append(params_raw)
    if not buckets:
        return True
    for bucket in buckets:
        reg = _str_val(bucket.get("doctor_registration_number"))
        name = _str_val(bucket.get("doctor_name"))
        if (not reg or reg.lower() == "present") or not _is_usable_doctor_name(name):
            return True
    return False


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
    category_hint: Optional[str] = None,
    extract_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    page_note = (
        " Multiple pages attached — read all pages and merge extracted fields."
        if len(image_blocks) > 1
        else ""
    )
    hint = normalize_document_name_hint(category_hint)
    field_list = tuple(extract_fields or ())
    hint_note = ""
    if hint:
        hint_note = f" This document is a '{hint}'."
    if field_list:
        targets = ", ".join(field_list)
        date_note = ""
        if "sample_collection_date" in field_list or "report_date" in field_list:
            date_note = (
                " DATE ALIASES: sample_collection_date = Registered On, Sample Collected On, "
                "Collected On, Received On, Scan Date, Examination Date "
                "(on radiology/HRCT, Registered On IS sample_collection_date). "
                "report_date = Reported On, Report Date, Date of Report. Use YYYY-MM-DD."
            )
        rx_note = ""
        if any(
            key in field_list
            for key in (
                "doctor_name",
                "doctor_registration_number",
                "doctor_signature",
                "prescribed_medicines",
            )
        ):
            rx_note = (
                " RX FOOTER: doctor_name = full printed name or \"\"; "
                "doctor_registration_number = Regn no / Reg No / WBMC/MCI (number only); "
                "doctor_signature = \"present\" if a handwritten signature is visible; "
                "prescribed_medicines = medicine table rows [{medicine, dosage}]."
            )
        user_text = (
            "Extract ONLY the caller-requested keys into parameters. "
            f"TARGET FIELDS: {targets}. "
            "Use \"\" or [] if a value is not printed. Do not invent values."
            + date_note
            + rx_note
            + page_note
            + hint_note
        )
        schema = _build_slim_document_schema(field_list)
        max_tokens = _openai_max_tokens_for_request(hint or "", field_list)
        system_prompt = FIELDS_SYSTEM_PROMPT
    else:
        if hint:
            hint_note = (
                f" CALLER LABEL: document_category MUST be '{hint}'. "
                f"Focus extraction on the matching parameters bucket for '{hint}', "
                "but still fill any other visible fields into their buckets. "
                "Do not reclassify away from the caller label."
            )
        user_text = (
            "Classify content type and extract ALL visible parameters into every matching "
            "bucket (prescription_parameters, invoice_parameters, report_parameters, "
            "payment_receipt_parameters). Do not skip a field because of document_category."
            + page_note
            + hint_note
            + (
                ""
                if hint
                else (
                    " Use document_category=payment_receipt for UPI/bank/card payment "
                    "screenshots. Use document_category=other only for selfies, blank "
                    "pages, or unrelated non-claim images."
                )
            )
        )
        schema = DOCUMENT_SCHEMA
        max_tokens = _openai_max_tokens_for_request(hint or "", None)
        system_prompt = SYSTEM_PROMPT
    return _call_openai_json(
        client,
        model,
        system_prompt,
        user_text,
        image_blocks,
        "medical_document_extraction",
        schema,
        max_tokens,
    )






def _apply_doctor_reg_from_refined(
    rx: Dict[str, Any], refined: Dict[str, Any]
) -> None:
    name = _str_val(refined.get("doctor_name"))
    if _is_usable_doctor_name(name):
        existing = _str_val(rx.get("doctor_name"))
        if not _is_usable_doctor_name(existing):
            rx["doctor_name"] = name
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
    """Vision pass on header/stamp crops when doctor name or registration is still empty."""
    stamp_blocks = build_stamp_blocks_from_document(doc.raw, doc=doc)
    header_blocks = build_gst_header_blocks_from_document(doc.raw, doc=doc)
    crop_blocks: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for block in header_blocks + stamp_blocks:
        url = str((block.get("image_url") or {}).get("url") or "")
        if url and url not in seen:
            seen.add(url)
            crop_blocks.append(block)
    if not crop_blocks:
        return

    refined = _call_openai_json(
        client,
        model,
        DOCTOR_REG_PROMPT,
        (
            "Zoomed letterhead and bottom / signature-zone crops of a prescription or OPD card. "
            "Find doctor_name on the letterhead (transliterate regional scripts to English) and "
            "CN No / Reg No on or under the doctor stamp or signature. Ignore CR No / Token / Mobile."
        ),
        crop_blocks,
        "doctor_registration_stamp_extraction",
        DOCTOR_REG_SCHEMA,
        350,
    )
    rx_raw = data.get("prescription_parameters")
    rx: Dict[str, Any] = dict(rx_raw) if isinstance(rx_raw, dict) else {}
    _apply_doctor_reg_from_refined(rx, refined)
    _normalize_doctor_registration(rx)
    data["prescription_parameters"] = rx
    params_raw = data.get("parameters")
    if isinstance(params_raw, dict):
        params = dict(params_raw)
        _apply_doctor_reg_from_refined(params, refined)
        _normalize_doctor_registration(params)
        data["parameters"] = params


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


def classify_document_url_openai(
    url: str,
    category_hint: Optional[str] = None,
    extract_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Hybrid classify: Textract OCR + OpenAI Vision in parallel, then merge.

    category_hint: optional CRM label (invoice / prescription / report / payment_receipt).
    extract_fields: optional subset of parameter keys — slimmer schema, faster extraction.
    Optional Vision GST/stamp refine is off by default (OPENAI_REFINE_PASSES).
    """
    started = time.perf_counter()
    model = (os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    client = get_openai_client()
    doc = load_document(url)
    image_blocks = build_vision_blocks_from_document(doc)
    t_load = time.perf_counter()
    hint = normalize_document_name_hint(category_hint)
    fields = normalize_extract_fields(extract_fields)
    run_textract = use_textract_for_category_hint(hint)

    ocr: Dict[str, str] = {}
    if run_textract:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_ocr = pool.submit(_run_textract_ocr, doc)
            fut_ai = pool.submit(
                _call_openai_vision,
                client,
                model,
                image_blocks,
                hint or None,
                fields,
            )
            ocr = fut_ocr.result() or {}
            data = fut_ai.result()
    else:
        data = _call_openai_vision(
            client, model, image_blocks, hint or None, fields
        )
    t_main = time.perf_counter()

    if hint:
        _apply_category_hint(data, hint)
    else:
        _recover_medical_classification(data)
    if ocr:
        merge_textract_into_openai_data(data, ocr)
        if hint:
            _apply_category_hint(data, hint)
        else:
            _recover_medical_classification(data)

    if doc.is_pdf:
        try:
            pdf_text = extract_pdf_text(doc.raw, max_pages=pdf_vision_max_pages())
        except Exception:
            logger.exception("PDF text extract failed for %s", url)
            pdf_text = ""
        if pdf_text:
            _fill_labeled_dates_from_text(data, pdf_text)

    refine_ms = 0.0
    if _OPENAI_REFINE_PASSES:
        t_refine0 = time.perf_counter()
        try:
            if _fields_need_gst_refine(fields) and (
                _peek_pharmacy_regulatory_needs_refine(data)
                or _peek_invoice_gst_needs_refine(data)
            ):
                _pause_between_openai_calls()
                _refine_gst_dl_if_needed(client, model, image_blocks, data, doc)
        except Exception:
            logger.exception("GST/DL header pass failed for %s", url)

        try:
            if _fields_need_doctor_reg_refine(fields) and _peek_doctor_registration_needs_refine(
                data
            ):
                _pause_between_openai_calls()
                _refine_doctor_registration_stamp(client, model, data, doc)
        except Exception:
            logger.exception("Doctor registration stamp pass failed for %s", url)
        refine_ms = (time.perf_counter() - t_refine0) * 1000.0

        if ocr:
            merge_textract_into_openai_data(data, ocr)
            if hint:
                _apply_category_hint(data, hint)

    result = _build_public_response(
        url, data, category_hint=hint or None, extract_fields=fields
    )
    total_ms = (time.perf_counter() - started) * 1000.0
    result["processing_time_ms"] = round(total_ms, 1)
    logger.info(
        "classify_document url=%s hint=%s textract=%s load_ms=%.0f main_ms=%.0f refine_ms=%.0f total_ms=%.0f",
        url.split("/")[-1],
        hint or "-",
        run_textract,
        (t_load - started) * 1000.0,
        (t_main - t_load) * 1000.0,
        refine_ms,
        total_ms,
    )
    return result


def _patient_match_key(raw: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", _str_val(raw).lower())


def _refresh_prescription_result_completeness(result: Dict[str, Any]) -> None:
    """Recompute completeness after post-batch doctor_name enrichment."""
    params = result.get("parameters")
    if not isinstance(params, dict):
        return
    fields = result.get("fields")
    if fields:
        required = tuple(fields)
        extra: Tuple[Any, ...] = ()
    else:
        subtype = str(result.get("prescription_subtype") or "opd")
        required = PRESCRIPTION_SUBTYPE_REQUIRED.get(subtype, PRESCRIPTION_REQUIRED)
        extra = _prescription_subtype_extra_checks(subtype, params)
    completeness, missing = _completeness(params, required, extra)
    result["completeness_percent"] = completeness
    result["missing_parameters"] = missing


def enrich_claim_batch_doctor_names(results: List[Dict[str, Any]]) -> None:
    """Fill weak prescription doctor_name from sibling claim documents (e.g. pharmacy invoice)."""
    by_patient: Dict[str, str] = {}
    fallback_names: List[str] = []

    for result in results:
        if result.get("error"):
            continue
        params = result.get("parameters")
        if not isinstance(params, dict):
            continue
        name = _str_val(params.get("doctor_name"))
        if not _is_usable_doctor_name(name):
            continue
        patient_key = _patient_match_key(params.get("patient_name"))
        if patient_key:
            by_patient.setdefault(patient_key, name)
        else:
            fallback_names.append(name)

    if not by_patient and not fallback_names:
        return

    for result in results:
        if result.get("error") or result.get("document_category") != "prescription":
            continue
        params = result.get("parameters")
        if not isinstance(params, dict):
            continue
        if _is_usable_doctor_name(params.get("doctor_name")):
            continue
        patient_key = _patient_match_key(params.get("patient_name"))
        candidate = by_patient.get(patient_key, "")
        if not candidate and len(fallback_names) == 1:
            candidate = fallback_names[0]
        if not candidate:
            continue
        params["doctor_name"] = candidate
        _normalize_doctor_name_value(params)
        if _is_usable_doctor_name(params.get("doctor_name")):
            _refresh_prescription_result_completeness(result)
