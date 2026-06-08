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

from openai import APIStatusError, RateLimitError

from app.core.openai_client import get_openai_client
from app.services.document_image_fetch import (
    DocumentPages,
    build_gst_header_blocks_from_document,
    build_header_crop_from_document,
    build_refine_image_blocks,
    build_regulatory_header_blocks,
    build_vision_blocks_from_document,
    load_document,
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

PRESCRIPTION_REQUIRED: Tuple[str, ...] = (
    "patient_name",
    "consultation_date",
    "clinic_hospital_name",
    "doctor_name",
    "doctor_registration_number",
    "doctor_signature",
    "doctor_stamp",
)

# Per Excel tab — only these keys are returned in the API for each prescription_subtype.
PRESCRIPTION_SUBTYPE_PARAM_KEYS: Dict[str, Tuple[str, ...]] = {
    "opd": (
        "patient_name",
        "patient_age",
        "patient_gender",
        "consultation_date",
        "clinic_hospital_name",
        "clinic_hospital_address",
        "doctor_name",
        "doctor_registration_number",
        "doctor_qualification",
        "doctor_signature",
        "doctor_stamp",
        "diagnosis",
        "presenting_complaints",
        "line_of_treatment",
        "prescribed_medicines",
        "advised_tests",
        "followup_date",
    ),
    "pharmacy": (
        "patient_age",
        "patient_gender",
        "consultation_date",
        "clinic_hospital_name",
        "clinic_hospital_address",
        "doctor_name",
        "doctor_registration_number",
        "doctor_qualification",
        "doctor_signature",
        "doctor_stamp",
        "diagnosis",
        "presenting_complaints",
        "line_of_treatment",
        "prescribed_medicines",
        "follow_up_advice",
    ),
    "diagnostic": (
        "patient_name",
        "patient_age",
        "patient_gender",
        "consultation_date",
        "clinic_hospital_name",
        "clinic_hospital_address",
        "doctor_name",
        "doctor_registration_number",
        "doctor_qualification",
        "doctor_signature",
        "doctor_stamp",
        "diagnosis",
        "presenting_complaints",
        "line_of_treatment",
        "advised_tests",
        "followup_date",
    ),
    "dental": (
        "patient_name",
        "patient_age",
        "patient_gender",
        "consultation_date",
        "clinic_hospital_name",
        "clinic_hospital_address",
        "doctor_name",
        "doctor_registration_number",
        "doctor_qualification",
        "doctor_signature",
        "doctor_stamp",
        "diagnosis",
        "affected_tooth_number",
        "treatment_advised",
        "treatment_plan",
        "procedure_recommendation",
    ),
    "eye_care": (
        "patient_name",
        "patient_age",
        "patient_gender",
        "consultation_date",
        "clinic_hospital_name",
        "clinic_hospital_address",
        "doctor_name",
        "doctor_registration_number",
        "doctor_qualification",
        "doctor_signature",
        "doctor_stamp",
        "diagnosis",
        "visual_acuity_details",
        "eye_power_prescription",
        "treatment_advice",
        "glasses_contact_lens_prescription",
        "follow_up_advice",
    ),
}

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

_PRESCRIPTION_REFINE_STRING_KEYS = tuple(
    key
    for key in PRESCRIPTION_PARAM_KEYS
    if key not in ("prescribed_medicines", "advised_tests")
)

_DOCTOR_REG_IN_TEXT_RE = re.compile(
    r"(?:RMC|MCI|MMC|reg\.?\s*no\.?|registration\s*no\.?)\s*[:\s#-]*([A-Za-z0-9][A-Za-z0-9/\-\s]{2,20})",
    re.IGNORECASE,
)
_DOCTOR_REG_DIGITS_RE = re.compile(r"\b(\d{4,6}(?:/\d{2,6})?)\b")

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
_REPORT_SECTION_HEADING_RE = re.compile(
    r"\b(?:examination|investigation|analysis)\s+of\s+\w+",
    re.IGNORECASE,
)
_REPORT_DOCUMENT_TITLE_RE = re.compile(
    r"^(?:clinical\s+)?laboratory(?:\s+report)?$|^(?:pathology|lab|radiology)\s+report$",
    re.IGNORECASE,
)
_REPORT_SPECIFIC_TEST_RE = re.compile(
    r"\b(?:sr\.?|serum|plasma|vit(?:amin)?|cbc|lft|kft|tsh|hba1c|"
    r"uric\s*acid|cholesterol|glucose|ha?emoglobin|bilirubin|creatinine|"
    r"platelet|wbc|rbc|esr|crp|psa|afp|sgot|sgpt|alt|ast)\b",
    re.IGNORECASE,
)

_INVOICE_REFINE_STRING_KEYS = (
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
    "consultation_charges",
    "registration_charges",
    "total_amount",
    "payment_mode",
    "transaction_reference",
    "gst_number",
    "drug_license_number",
    "authorized_stamp",
    "authorized_signature",
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

SYSTEM_PROMPT = """You are a medical document analyst for Indian healthcare insurance claims.

CONTENT (filled-in data only — patient name, dates, tests, medicines, amounts, signatures; ignore blank pre-printed form rows):
- document_type = handwritten if content is mostly written by hand.
- document_type = computer_generated if content is fully typed/printed digitally (no handwriting in filled fields).
- uncertain if unclear.

content_handwritten_percent / content_computer_generated_percent = split of filled content only. Must sum to 100.
Cash memo with handwritten patient/tests/prices → document_type=handwritten, content_handwritten_percent ~100.
Printed tax invoice, retail invoice, bill of supply, or POS pharmacy receipt → document_type=computer_generated,
content_computer_generated_percent 90-100 (only pharmacist signature/stamp at footer may be handwritten ~5-10%).
Do NOT mark fully printed pharmacy receipts as handwritten because of a signature at the bottom.

document_category:
- prescription | invoice | report — valid medical claim documents for insurance.
- pharmacy/chemist Bill of Supply, tax invoice, cash memo from medical store ARE invoices (is_medical_document=true, invoice_subtype=pharmacy).
- diagnostic center bills → invoice_subtype=diagnostic.
- Doctor/clinic consultation receipts ("Received with thanks", OPD fee, cash memo from clinic) → invoice, invoice_subtype=opd_consultation.
- Dental clinic bills → invoice_subtype=dental. Optical / eye care / lens / frame bills → invoice_subtype=eye_care.
- other — ONLY for non-documents: app screenshots, error popups, phone UI, selfies, blank/unreadable images.
  Do NOT mark pharmacy bills, clinic receipts, or hospital bills as other.

Orientation: photos may be upside-down or sideways — still read and classify. Rotated clinic/bill/Rx photos ARE medical documents.

is_medical_document=true for Rx, pharmacy bill, diagnostic bill, lab report, doctor consultation receipt.

Rx without rupee total → prescription, NEVER invoice.

invoice_subtype (invoice only): pharmacy | diagnostic | opd_consultation | dental | eye_care | uncertain | not_applicable.

Invoice common fields: patient_name, patient_age, patient_gender (when shown), invoice_number, invoice_date,
provider_name, provider_address, provider_contact, doctor_name, total_amount, payment_mode (Cash/UPI/Card/Online),
transaction_reference (payment ref / UPI ref if shown), authorized_stamp, authorized_signature.

ALL invoice types (pharmacy, diagnostic, OPD, dental, eye care): if GSTIN / GST NO. is printed anywhere on the
bill header or letterhead, you MUST fill gst_number (15-char code only, no label). Scan the full top margin —
GST is often small text in the top-right corner. Use "" only when no GST number is printed on the document.

OPD consultation invoice: consultation_charges, registration_charges (if separate), service_details[] (e.g. "Consultation", "Follow-up").
Diagnostic invoice: sample_collection_date, test_details[] as "Test name — Rs amount" per line.
Dental invoice: item_details[] (procedure/treatment lines with amount if shown), service_details[] for consultation if any.
Eye care / optical invoice: item_details[] (frame, lens type Single Vision/Bifocal/Progressive, contact lens, qty, rate, amount per line).

Invoice / pharmacy bill: medicine_details[] — one string per medicine row with product name & packing; include Qty, Rate, Batch, Expiry when printed
(e.g. "FAROVIA-200 TAB 1*6 | Qty 20 | Rate 680 | Batch JDKAA04 | Exp 10/26").
Pharmacy / chemist / Bill of Supply: read the TOP header for gst_number (15-char GSTIN only — no "GSTIN:" prefix)
and drug_license_number (all DL NO. lines exactly as printed; formats vary by state — join multiple with "; ").
total_amount: numeric value only from the primary total line on that bill (Total MRP Value, Grand Total, Invoice Value,
or Amount — whichever is the main total for that format). Copy digits exactly; do not calculate. Use "" if unreadable, never "N/A".
Pharmacy footer: scan bottom-right for authorized_stamp (shop/pharmacist/proprietor rubber stamp text) and
authorized_signature (handwritten sign). Use "present" when visible but illegible. Fill at least one when shown on bill.
prescription_subtype (prescription only): opd | pharmacy | diagnostic | dental | eye_care | uncertain | not_applicable.

Prescription common: patient_name, patient_age, patient_gender, consultation_date, clinic_hospital_name, clinic_hospital_address
(letterhead), doctor_name, doctor_qualification (MBBS/MD/DNB etc.), doctor_registration_number, doctor_signature, doctor_stamp,
diagnosis, presenting_complaints, line_of_treatment, treatment_plan, followup_date, follow_up_advice.

OPD Rx (prescription_subtype=opd): prescribed_medicines[{medicine,dosage}] with frequency & duration in dosage; advised_tests if labs ordered.
Pharmacy Rx (prescription_subtype=pharmacy): focus on prescribed_medicines with complete names and dosage instructions.
Diagnostics Rx (prescription_subtype=diagnostic): advised_tests[] only when labs ordered; diagnosis and complaints.
Dental Rx (prescription_subtype=dental): affected_tooth_number, treatment_advised, procedure_recommendation (Filling/Extraction/RCT/Scaling etc.).
Eye care Rx (prescription_subtype=eye_care): visual_acuity_details, eye_power_prescription (SPH/CYL/AXIS/Add), treatment_advice,
glasses_contact_lens_prescription when applicable.

doctor_registration_number: read from rubber stamp, letterhead, or printed text (RMC No., Reg No., MCI, MMC). NOT empty if visible.
doctor_stamp / doctor_signature: "present" if visible but illegible.

Report (lab/radiology):
- Formal lab report with TYPED/PRINTED patient, date, test name, numeric result, reference range → document_type=computer_generated, content_computer_generated_percent 95-100 (only signature may be handwritten ~0-10%).
- Do NOT mark printed lab reports as handwritten just because letterhead is printed.
- test_names: specific test (e.g. "Sr. Uric Acid"), not section title alone ("EXAMINATION OF BLOOD").
- pathologist_signature: "present" if handwritten signature visible.
"""

def _prescription_refine_schema() -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "advised_tests": _STRING_ARRAY,
        "prescribed_medicines": {"type": "array", "items": _MEDICINE_ITEM_SCHEMA},
    }
    for key in _PRESCRIPTION_REFINE_STRING_KEYS:
        props[key] = {"type": "string"}
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


PRESCRIPTION_REFINE_SCHEMA = _prescription_refine_schema()

# Backward-compatible alias
PRESCRIPTION_MEDICINE_SCHEMA = PRESCRIPTION_REFINE_SCHEMA

DIAGNOSTIC_INVOICE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": ["handwritten", "computer_generated", "uncertain"],
        },
        "content_handwritten_percent": {"type": "number"},
        "content_computer_generated_percent": {"type": "number"},
        "patient_name": {"type": "string"},
        "patient_age": {"type": "string"},
        "patient_gender": {"type": "string"},
        "doctor_name": {"type": "string"},
        "invoice_number": {"type": "string"},
        "invoice_date": {"type": "string"},
        "sample_collection_date": {"type": "string"},
        "total_amount": {"type": "string"},
        "payment_mode": {"type": "string"},
        "transaction_reference": {"type": "string"},
        "provider_name": {"type": "string"},
        "provider_address": {"type": "string"},
        "provider_contact": {"type": "string"},
        "consultation_charges": {"type": "string"},
        "registration_charges": {"type": "string"},
        "gst_number": {"type": "string"},
        "drug_license_number": {"type": "string"},
        "test_details": _STRING_ARRAY,
        "medicine_details": _STRING_ARRAY,
        "service_details": _STRING_ARRAY,
        "item_details": _STRING_ARRAY,
        "authorized_stamp": {"type": "string"},
        "authorized_signature": {"type": "string"},
    },
    "required": [
        "document_type",
        "content_handwritten_percent",
        "content_computer_generated_percent",
        "patient_name",
        "patient_age",
        "patient_gender",
        "doctor_name",
        "invoice_number",
        "invoice_date",
        "sample_collection_date",
        "total_amount",
        "payment_mode",
        "transaction_reference",
        "provider_name",
        "provider_address",
        "provider_contact",
        "consultation_charges",
        "registration_charges",
        "gst_number",
        "drug_license_number",
        "test_details",
        "medicine_details",
        "service_details",
        "item_details",
        "authorized_stamp",
        "authorized_signature",
    ],
    "additionalProperties": False,
}

INVOICE_REFINE_PROMPT = """Medical bill image — pharmacy, diagnostic lab, OPD/clinic, dental, or eye care / optical invoice.

These ARE valid medical claim documents (is_medical_document=true, document_category=invoice).

CONTENT: document_type and content_handwritten_percent / content_computer_generated_percent (sum 100).
Pre-printed Cash Memo with handwritten patient name, date, medicine lines, qty/rate/amount, totals →
document_type=handwritten, content_handwritten_percent 85-100 (only letterhead is printed).
Printed tax invoice / retail invoice / bill of supply / POS receipt with typed line items → computer_generated;
only footer signature may be handwritten (~5-10%).

Common: patient_name, patient_age, patient_gender, invoice_number, invoice_date, provider_name, provider_address,
provider_contact, doctor_name, total_amount, payment_mode, transaction_reference, authorized_stamp,
authorized_signature ("present" if illegible).

gst_number — REQUIRED when printed on ANY invoice type (pharmacy, diagnostic lab, OPD/clinic, dental, eye care).
Read the top header / letterhead for the 15-character GSTIN (GST NO., GSTIN, GSTIN/UIN). Return code only, no label.
The 13th character is always letter Z. Scan top-right margin — often small font. Use "" only if not printed.

OPD / clinic consultation (invoice_subtype=opd_consultation):
- consultation_charges, registration_charges (separate if printed), service_details[] (Consultation / Follow-up)
- doctor_name, clinic/hospital letterhead in provider_name / authorized_stamp

Pharmacy / Bill of Supply (invoice_subtype=pharmacy):
- gst_number (15-char GSTIN only), drug_license_number (all DL NO. lines; join with "; ")
- medicine_details[]: product | Qty | Rate | Batch | Expiry per row when printed
- authorized_stamp / authorized_signature at footer

Diagnostic centre (invoice_subtype=diagnostic):
- sample_collection_date, test_details[] as "Test name — Rs amount" per line
- authorized_stamp = lab seal; authorized_signature = signatory or "present"

Dental (invoice_subtype=dental): item_details[] for procedures; provider_name = dental clinic.

Eye care / optical (invoice_subtype=eye_care): item_details[] for frame, lens (Single Vision/Bifocal/Progressive),
contact lens, qty, rate, amount per line; provider_name = optical store or eye clinic."""

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

INVOICE_GST_PROMPT = """Indian medical invoice / bill / cash memo / tax invoice image.

Extract ONLY gst_number from the TOP header, letterhead, and top margin:

gst_number — 15-character GSTIN (labels: GST NO., GSTIN, GSTIN/UIN). Return the code ONLY with no label.
GSTIN format: 2 digits, 5 letters, 4 digits, 1 letter, 1 letter/digit, then always letter Z, then 1 letter/digit.
The 13th character is ALWAYS the letter Z — do not confuse it with digit 2.
Applies to pharmacy bills, diagnostic lab bills, clinic/OPD receipts, dental and eye-care invoices.
Do NOT return FSSAI number as GSTIN. Scan the full width of the top area including top-right corner.
Use "" only if no GST number is genuinely printed on the document."""

PHARMACY_REGULATORY_PROMPT = """Indian pharmacy / chemist / medical store bill image.

Extract ONLY regulatory identifiers from the TOP header, letterhead, and top margin (small text above the shop name):

gst_number — 15-character GSTIN (labels: GST NO., GSTIN, GSTIN/UIN). Return the code ONLY with no label.
GSTIN format: 2 digits, 5 letters, 4 digits, 1 letter, 1 letter/digit, then always letter Z, then 1 letter/digit.
The 13th character is ALWAYS the letter Z — do not confuse it with digit 2. Example: 14AMQPD0832R2Z6.
Do NOT return FSSAI number as GSTIN.

drug_license_number — EVERY drug license on the bill (DL NO., DL No.20, DL No.21, Form 20, Form 21).
Copy each exactly as printed. Join multiple with "; ".
Formats vary: 20-466324, 21-466325 (comma-separated pairs), 20-DRUG/2019-20/34132, 27/MD/MAH/000005.

Both fields are usually in the top header — extract BOTH when visible. Do not stop after finding one.
Scan the full width of the top area — GST and DL are often small font near the store name.
Use "" only if that field is genuinely not printed anywhere on the bill."""

_REPORT_REFINE_STRING_KEYS = (
    "patient_name",
    "patient_age",
    "patient_gender",
    "laboratory_name",
    "laboratory_address",
    "sample_collection_date",
    "report_date",
    "pathologist_name",
    "pathologist_registration_number",
    "pathologist_signature",
    "authorized_stamp",
)

LAB_REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": ["handwritten", "computer_generated", "uncertain"],
        },
        "content_handwritten_percent": {"type": "number"},
        "content_computer_generated_percent": {"type": "number"},
        "patient_name": {"type": "string"},
        "report_date": {"type": "string"},
        "sample_collection_date": {"type": "string"},
        "test_names": _STRING_ARRAY,
        "test_results": _STRING_ARRAY,
        "reference_ranges": _STRING_ARRAY,
        "pathologist_name": {"type": "string"},
        "pathologist_registration_number": {"type": "string"},
        "pathologist_signature": {"type": "string"},
    },
    "required": [
        "document_type",
        "content_handwritten_percent",
        "content_computer_generated_percent",
        "patient_name",
        "report_date",
        "sample_collection_date",
        "test_names",
        "test_results",
        "reference_ranges",
        "pathologist_name",
        "pathologist_registration_number",
        "pathologist_signature",
    ],
    "additionalProperties": False,
}

LAB_REPORT_PROMPT = """Clinical laboratory / diagnostic REPORT image.

CONTENT type (filled data only — ignore letterhead/layout):
- computer_generated: patient, dates, test names, numeric results, reference ranges are PRINTED/TYPED in a uniform font. Use this for standard lab printouts.
- handwritten: those fields are written by hand (unusual for lab reports).
- If only the pathologist signature at the bottom is handwritten → document_type=computer_generated, content_handwritten_percent=0 to 10, content_computer_generated_percent=90 to 100.

Extract test_names as the specific investigation (e.g. "Sr. Uric Acid"), NOT section headers like "EXAMINATION OF BLOOD" alone.
test_results and reference_ranges as parallel array entries per test."""

PRESCRIPTION_MEDICINE_PROMPT = """Extract from this prescription (Rx) image — OPD, pharmacy, diagnostic, dental, or eye care Rx.

1) prescribed_medicines — EVERY drug line {medicine, dosage}; dosage must include frequency and duration when shown.

2) advised_tests — lab/diagnostic tests only (CBC, MRI, X-ray, etc.). NOT medicines. Empty array if none.

3) Patient & clinic: patient_name, patient_age, patient_gender, consultation_date, clinic_hospital_name, clinic_hospital_address.

4) Doctor: doctor_name, doctor_qualification, doctor_registration_number (from stamp/letterhead — RMC/MCI/MMC/Reg No.),
   doctor_stamp (text or "present"), doctor_signature ("present" if illegible).

5) Clinical: diagnosis, presenting_complaints, line_of_treatment, treatment_plan, followup_date, follow_up_advice.

6) Dental only: affected_tooth_number, treatment_advised, procedure_recommendation (Filling/Extraction/RCT/Scaling).

7) Eye care only: visual_acuity_details, eye_power_prescription (SPH/CYL/AXIS/Add), treatment_advice,
   glasses_contact_lens_prescription."""

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
    """Fill doctor_registration_number from stamp text when the model only marked stamp present."""
    reg = _str_val(params.get("doctor_registration_number"))
    if reg and reg.lower() != "present":
        params["doctor_registration_number"] = reg
        return

    stamp = _str_val(params.get("doctor_stamp"))
    sources = [stamp, _str_val(params.get("doctor_name"))]
    for source in sources:
        if not source:
            continue
        match = _DOCTOR_REG_IN_TEXT_RE.search(source)
        if match:
            params["doctor_registration_number"] = match.group(1).strip()
            return
        digit_match = _DOCTOR_REG_DIGITS_RE.search(source)
        if digit_match and stamp and stamp.lower() != "present":
            params["doctor_registration_number"] = digit_match.group(1)
            return

    if stamp and stamp.lower() != "present":
        digit_match = _DOCTOR_REG_DIGITS_RE.search(stamp)
        if digit_match:
            params["doctor_registration_number"] = digit_match.group(1)


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
    compact = re.sub(r"\s+", "", text.upper())
    # Strip label prefixes — replace with separator char so adjacent chars don't merge
    compact = re.sub(r"GSTIN(?:/UIN)?[:/]?|GSTNO[./:]?", "|", compact)
    compact = re.sub(r"\|+", "|", compact)
    match = _GSTIN_RE.search(compact)
    if match:
        return match.group(1).upper()
    # Fuzzy recovery: position 13 (0-indexed) must be Z; fix if model misread it
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


def _pharmacy_invoice_needs_refine(inv: Dict[str, Any]) -> bool:
    return _pharmacy_regulatory_incomplete(inv) or _invoice_authorization_missing(inv)


def _total_amount_needs_cleanup(inv: Dict[str, Any]) -> bool:
    raw = _str_val(inv.get("total_amount"))
    if not raw:
        return False
    clean = raw.replace(",", "").strip()
    return not re.fullmatch(r"\d+\.\d{2}", clean) and not re.fullmatch(r"\d+", clean)


def _pharmacy_extraction_needs_refine(inv: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """Any pharmacy bill missing key fields or misclassified as handwritten."""
    if not _looks_like_pharmacy_invoice(inv, data):
        return False
    if _pharmacy_invoice_needs_refine(inv):
        return True
    if _looks_like_structured_printed_invoice(inv) and str(
        data.get("document_type")
    ) == "handwritten":
        return True
    if _total_amount_needs_cleanup(inv):
        return True
    return False


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


def _prescription_keys_for_subtype(
    subtype: str, params: Optional[Dict[str, Any]] = None
) -> Tuple[str, ...]:
    if subtype in PRESCRIPTION_SUBTYPE_PARAM_KEYS:
        return PRESCRIPTION_SUBTYPE_PARAM_KEYS[subtype]
    if params:
        filled = tuple(
            key for key in PRESCRIPTION_PARAM_KEYS if _is_filled(params, key)
        )
        if filled:
            return filled
    return PRESCRIPTION_SUBTYPE_PARAM_KEYS["opd"]


def _filter_params_by_keys(
    params: Dict[str, Any],
    keys: Sequence[str],
    array_keys: frozenset[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in keys:
        if key in array_keys:
            result[key] = _list_val(params.get(key))
        elif key == "prescribed_medicines":
            result[key] = _normalize_prescribed_medicines(params.get(key))
        else:
            result[key] = _str_val(params.get(key))
    return result


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


def _looks_like_report_section_heading(name: str) -> bool:
    """Broad section title on lab report — not a specific test/analyte name."""
    text = name.strip()
    if not text:
        return True
    if _REPORT_SPECIFIC_TEST_RE.search(text):
        return False
    if _REPORT_SECTION_HEADING_RE.search(text):
        return True
    if _REPORT_DOCUMENT_TITLE_RE.match(text):
        return True
    low = text.lower()
    if text.isupper() and len(text.split()) <= 6:
        if any(kw in low for kw in ("examination", "investigation", "panel", "profile")):
            return True
    return False


def _is_report_section_header_only(test_names: List[str]) -> bool:
    if not test_names:
        return False
    return all(_looks_like_report_section_heading(n) for n in test_names)


def _report_extraction_needs_refine(report_params: Dict[str, Any]) -> bool:
    """Trigger refine when test_names look like headings or don't match results."""
    names = _list_val(report_params.get("test_names"))
    results = _list_val(report_params.get("test_results"))
    ranges = _list_val(report_params.get("reference_ranges"))
    if not names:
        return True
    if _is_report_section_header_only(names):
        return True
    if results and len(results) > len(names):
        return True
    if ranges and len(ranges) > len(names):
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


def _medical_extraction_score(data: Dict[str, Any]) -> int:
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
    return max(
        _invoice_extraction_score(inv),
        _count_extracted_fields(rx),
        _count_extracted_fields(rep),
    )


def _needs_orientation_retry(data: Dict[str, Any]) -> bool:
    """Re-classify rotated 180° when first pass returned other / empty."""
    category = str(data.get("document_category", "other"))
    if category == "other":
        return True
    if not data.get("is_medical_document", True) and _all_medical_blocks_empty(data):
        return True
    if _all_medical_blocks_empty(data) and category in ("invoice", "prescription", "report"):
        return True
    return False


def _pick_better_classification(
    original: Dict[str, Any], rotated: Dict[str, Any]
) -> Tuple[Dict[str, Any], bool]:
    """Return (best_data, used_rotated)."""
    score_orig = _medical_extraction_score(original)
    score_rot = _medical_extraction_score(rotated)
    cat_orig = str(original.get("document_category", "other"))
    cat_rot = str(rotated.get("document_category", "other"))

    if score_rot > score_orig:
        return rotated, True
    if score_rot < score_orig:
        return original, False
    if cat_orig == "other" and cat_rot != "other":
        return rotated, True
    if cat_rot == "other" and cat_orig != "other":
        return original, False
    if rotated.get("is_medical_document") and not original.get("is_medical_document"):
        return rotated, True
    return original, False


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
            "parameters": {},
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
        subtype_keys = _prescription_keys_for_subtype(prescription_subtype, parameters)
        parameters = _filter_params_by_keys(
            parameters, subtype_keys, _PRESCRIPTION_ARRAY_KEYS
        )
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
        "parameters": parameters,
        "missing_parameters": missing_parameters,
        "message": "",
    }


def _peek_prescription_category(data: Dict[str, Any]) -> str:
    category = str(data.get("document_category", "other"))
    if category == "other" or not data.get("is_medical_document", True):
        return "other"
    if _all_medical_blocks_empty(data):
        return "other"

    doc_type = str(data.get("document_type", "uncertain"))
    inv_raw = data.get("invoice_parameters")
    inv = inv_raw if isinstance(inv_raw, dict) else {}
    total = _str_val(inv.get("total_amount"))
    if category == "invoice" and not total:
        if doc_type == "handwritten":
            return "prescription"
        if doc_type == "computer_generated" and not (
            _is_filled(inv, "test_details") or _is_filled(inv, "medicine_details")
        ):
            if _count_extracted_fields(
                _normalize_params(
                    data.get("report_parameters"),
                    REPORT_PARAM_KEYS,
                    _REPORT_ARRAY_KEYS,
                )
            ):
                return "report"
            return "other"
    if category in _VALID_CATEGORIES:
        return category
    return "other"


def _merge_prescription_medicines(data: Dict[str, Any], refined: Dict[str, Any]) -> None:
    rx_raw = data.get("prescription_parameters")
    rx: Dict[str, Any] = dict(rx_raw) if isinstance(rx_raw, dict) else {}
    refined_meds = _normalize_prescribed_medicines(refined.get("prescribed_medicines"))
    refined_tests = _list_val(refined.get("advised_tests"))
    if refined_meds:
        rx["prescribed_medicines"] = refined_meds
    if refined_tests:
        rx["advised_tests"] = refined_tests
    for key in _PRESCRIPTION_REFINE_STRING_KEYS:
        _merge_string_field(rx, key, _str_val(refined.get(key)))
    _normalize_doctor_registration(rx)
    data["prescription_parameters"] = rx


def _merge_diagnostic_invoice(data: Dict[str, Any], refined: Dict[str, Any]) -> None:
    inv_raw = data.get("invoice_parameters")
    inv: Dict[str, Any] = dict(inv_raw) if isinstance(inv_raw, dict) else {}

    for key in _INVOICE_REFINE_STRING_KEYS:
        if key in ("gst_number", "drug_license_number"):
            continue
        _merge_string_field(inv, key, _str_val(refined.get(key)))
    _apply_regulatory_from_refined(inv, refined)

    tests = _normalize_invoice_detail_list(refined.get("test_details"))
    medicines = _normalize_invoice_detail_list(refined.get("medicine_details"))
    items = _normalize_invoice_detail_list(refined.get("item_details"))
    services = [
        s
        for s in _list_val(refined.get("service_details"))
        if not _is_generic_invoice_service_line(s)
    ]
    if tests:
        inv["test_details"] = tests
    if medicines:
        inv["medicine_details"] = medicines
    if items:
        inv["item_details"] = items
    if services:
        inv["service_details"] = services
    elif tests:
        inv["service_details"] = [
            s
            for s in _list_val(inv.get("service_details"))
            if not _is_generic_invoice_service_line(s)
        ]

    _normalize_invoice_fields(inv)
    data["invoice_parameters"] = inv

    if _str_val(refined.get("document_type")):
        data["document_type"] = refined["document_type"]
    if refined.get("content_handwritten_percent") is not None:
        data["content_handwritten_percent"] = refined["content_handwritten_percent"]
    if refined.get("content_computer_generated_percent") is not None:
        data["content_computer_generated_percent"] = refined["content_computer_generated_percent"]

    if medicines or _is_filled(inv, "drug_license_number") or _is_filled(inv, "gst_number"):
        data["invoice_subtype"] = "pharmacy"
    elif tests and not medicines:
        data["invoice_subtype"] = "diagnostic"
    elif items:
        data["invoice_subtype"] = _infer_invoice_subtype(inv, str(data.get("invoice_subtype", "")))
    elif _is_filled(inv, "consultation_charges") or _is_filled(inv, "registration_charges"):
        data["invoice_subtype"] = "opd_consultation"


def _merge_lab_report(data: Dict[str, Any], refined: Dict[str, Any]) -> None:
    rep_raw = data.get("report_parameters")
    rep: Dict[str, Any] = dict(rep_raw) if isinstance(rep_raw, dict) else {}

    for key in _REPORT_REFINE_STRING_KEYS:
        _merge_string_field(rep, key, _str_val(refined.get(key)))

    for key in ("test_names", "test_results", "reference_ranges"):
        items = _list_val(refined.get(key))
        if items:
            rep[key] = items

    data["report_parameters"] = rep

    if _str_val(refined.get("document_type")):
        data["document_type"] = refined["document_type"]
    if refined.get("content_handwritten_percent") is not None:
        data["content_handwritten_percent"] = refined["content_handwritten_percent"]
    if refined.get("content_computer_generated_percent") is not None:
        data["content_computer_generated_percent"] = refined["content_computer_generated_percent"]


def _peek_report_needs_refine(data: Dict[str, Any]) -> bool:
    if str(data.get("document_category")) != "report":
        return False
    if not data.get("is_medical_document", True) or _all_medical_blocks_empty(data):
        return False
    rep_raw = data.get("report_parameters")
    rep = rep_raw if isinstance(rep_raw, dict) else {}
    if str(data.get("document_type")) == "handwritten" and float(
        data.get("content_handwritten_percent", 0)
    ) >= 50:
        return True
    if _report_extraction_needs_refine(rep):
        return True
    if _looks_like_typed_lab_report(rep) and not _str_val(rep.get("pathologist_signature")):
        return True
    return False


def _peek_invoice_needs_refine(data: Dict[str, Any]) -> bool:
    if _non_medical_reason_suggests_bill(data) and _all_medical_blocks_empty(data):
        return True
    if str(data.get("document_category")) != "invoice":
        inv = _normalize_params(
            data.get("invoice_parameters"), INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS
        )
        if _invoice_extraction_score(inv) < 2:
            return False
    if not data.get("is_medical_document", True) and not _non_medical_reason_suggests_bill(data):
        return False
    inv_raw = data.get("invoice_parameters")
    inv = dict(inv_raw) if isinstance(inv_raw, dict) else {}
    _normalize_invoice_fields(inv)
    if not _is_filled(inv, "patient_name"):
        return True
    if _pharmacy_extraction_needs_refine(inv, data):
        return True
    tests = _normalize_invoice_detail_list(inv.get("test_details"))
    medicines = _normalize_invoice_detail_list(inv.get("medicine_details"))
    if medicines and len(medicines) >= 2:
        if _pharmacy_regulatory_invalid(inv) or _invoice_authorization_missing(inv):
            return True
        return False
    if medicines:
        return True
    if len(tests) < 2:
        return True
    services = _list_val(inv.get("service_details"))
    if services and all(_is_generic_invoice_service_line(s) for s in services):
        return True
    if str(data.get("document_type")) == "computer_generated" and float(
        data.get("content_handwritten_percent", 0)
    ) < 15:
        return True
    return False


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


def _call_openai_vision_rotated(
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
            "The document photo may be upside-down or sideways — read text in any orientation."
            + " Classify content type and extract parameters."
            + page_note
            + " Clinic/doctor consultation receipts are invoice (opd_consultation)."
            + " Fill only the matching parameter object."
        ),
        image_blocks,
        "medical_document_extraction_rotated",
        DOCUMENT_SCHEMA,
        2200,
    )


def _refine_prescription_medicines(
    client: Any,
    model: str,
    image_blocks: List[Dict[str, Any]],
    data: Dict[str, Any],
    document_raw: bytes | None = None,
    doc: DocumentPages | None = None,
) -> None:
    rx_model = (os.getenv("OPENAI_RX_MODEL") or model).strip()
    refine_blocks = build_refine_image_blocks(
        image_blocks, document_raw or (doc.raw if doc else b""), detail="high", doc=doc
    )
    refined = _call_openai_json(
        client,
        rx_model,
        PRESCRIPTION_MEDICINE_PROMPT,
        "List medicines, advised tests, and doctor registration number from the rubber stamp.",
        refine_blocks,
        "prescription_refine_extraction",
        PRESCRIPTION_REFINE_SCHEMA,
        2500,
    )
    _merge_prescription_medicines(data, refined)


def _invoice_refine_blocks(
    image_blocks: List[Dict[str, Any]],
    document_raw: bytes,
    data: Dict[str, Any],
    doc: DocumentPages | None,
) -> List[Dict[str, Any]]:
    blocks = build_refine_image_blocks(image_blocks, document_raw, detail="high", doc=doc)
    if not blocks:
        blocks = []
        for block in image_blocks:
            if block.get("type") != "image_url":
                blocks.append(block)
                continue
            image_url = block.get("image_url")
            if not isinstance(image_url, dict):
                blocks.append(block)
                continue
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {**image_url, "detail": "high"},
                }
            )
    blocks.extend(build_gst_header_blocks_from_document(document_raw, doc=doc))
    return blocks


def _refine_diagnostic_invoice(
    client: Any,
    model: str,
    image_blocks: List[Dict[str, Any]],
    data: Dict[str, Any],
    document_raw: bytes | None = None,
    doc: DocumentPages | None = None,
) -> None:
    inv_model = (os.getenv("OPENAI_INVOICE_MODEL") or model).strip()
    raw = document_raw or (doc.raw if doc else b"")
    refine_blocks = _invoice_refine_blocks(image_blocks, raw, data, doc)
    refined = _call_openai_json(
        client,
        inv_model,
        INVOICE_REFINE_PROMPT,
        (
            "Extract invoice fields for any medical bill type. Printed tax invoice / bill of supply = "
            "computer_generated. Read top header for gst_number (15-char GSTIN, no label) on ALL invoice "
            "types when printed. Pharmacy: every DL NO. line exactly. total_amount = numeric primary total "
            "as printed. Scan footer for stamp/signature."
        ),
        refine_blocks,
        "invoice_refine_extraction",
        DIAGNOSTIC_INVOICE_SCHEMA,
        1800,
    )
    _merge_diagnostic_invoice(data, refined)
    data["is_medical_document"] = True
    data["document_category"] = "invoice"
    _recover_medical_classification(data)


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


def _refine_pharmacy_gst_header(
    client: Any,
    model: str,
    document_raw: bytes,
    data: Dict[str, Any],
    doc: DocumentPages | None = None,
) -> None:
    _refine_invoice_gst_header(client, model, document_raw, data, doc=doc)


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


def _refine_lab_report(
    client: Any,
    model: str,
    image_blocks: List[Dict[str, Any]],
    data: Dict[str, Any],
    document_raw: bytes | None = None,
    doc: DocumentPages | None = None,
) -> None:
    report_model = (os.getenv("OPENAI_REPORT_MODEL") or model).strip()
    refine_blocks = build_refine_image_blocks(
        image_blocks, document_raw or (doc.raw if doc else b""), detail="high", doc=doc
    )
    refined = _call_openai_json(
        client,
        report_model,
        LAB_REPORT_PROMPT,
        "Classify printed vs handwritten content and extract lab test fields.",
        refine_blocks,
        "lab_report_extraction",
        LAB_REPORT_SCHEMA,
        2000,
    )
    _merge_lab_report(data, refined)


def classify_document_url_openai(url: str) -> Dict[str, Any]:
    """Classify one image URL and return category-specific extracted parameters."""
    model = (os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    client = get_openai_client()
    doc = load_document(url)
    image_blocks = build_vision_blocks_from_document(doc)
    data = _call_openai_vision(client, model, image_blocks)
    _recover_medical_classification(data)

    used_rotated = False
    if _needs_orientation_retry(data) and not doc.is_pdf and doc.page_images:
        try:
            _pause_between_openai_calls()
            doc_rot = doc.with_rotation(180)
            blocks_rot = build_vision_blocks_from_document(doc_rot)
            data_rot = _call_openai_vision_rotated(client, model, blocks_rot)
            _recover_medical_classification(data_rot)
            data, used_rotated = _pick_better_classification(data, data_rot)
            if used_rotated:
                doc = doc_rot
                image_blocks = blocks_rot
        except Exception:
            logger.exception("Orientation retry failed for %s", url)

    if _peek_invoice_needs_refine(data):
        try:
            _pause_between_openai_calls()
            _refine_diagnostic_invoice(client, model, image_blocks, data, doc.raw, doc)
        except Exception:
            logger.exception("Invoice refine pass failed for %s", url)

    if _peek_pharmacy_regulatory_needs_refine(data):
        try:
            _pause_between_openai_calls()
            inv_raw = data.get("invoice_parameters")
            inv = dict(inv_raw) if isinstance(inv_raw, dict) else {}
            _normalize_invoice_fields(inv)
            if _pharmacy_gst_only_missing(inv):
                _refine_invoice_gst_header(client, model, doc.raw, data, doc)
            else:
                _refine_pharmacy_regulatory(
                    client, model, image_blocks, data, doc.raw, doc
                )
        except Exception:
            logger.exception("Pharmacy regulatory refine pass failed for %s", url)
    elif _peek_invoice_gst_needs_refine(data):
        try:
            _pause_between_openai_calls()
            _refine_invoice_gst_header(client, model, doc.raw, data, doc)
        except Exception:
            logger.exception("Invoice GST header pass failed for %s", url)
    elif _peek_prescription_category(data) == "prescription":
        try:
            _pause_between_openai_calls()
            _refine_prescription_medicines(
                client, model, image_blocks, data, doc.raw, doc
            )
        except Exception:
            logger.exception("Prescription medicine refine pass failed for %s", url)
    elif _peek_report_needs_refine(data):
        try:
            _pause_between_openai_calls()
            _refine_lab_report(client, model, image_blocks, data, doc.raw, doc)
        except Exception:
            logger.exception("Lab report refine pass failed for %s", url)

    if _peek_invoice_gst_needs_refine(data):
        try:
            _pause_between_openai_calls()
            _refine_invoice_gst_header(client, model, doc.raw, data, doc)
        except Exception:
            logger.exception("Final invoice GST header pass failed for %s", url)

    return _build_public_response(url, data)
