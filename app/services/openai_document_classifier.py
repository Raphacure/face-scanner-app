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
from typing import Any, Dict, List, Sequence, Tuple

from openai import APIStatusError, RateLimitError

from app.core.openai_client import get_openai_client
from app.services.document_image_fetch import build_vision_image_blocks

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_RETRIES = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "6")))
OPENAI_INTER_CALL_DELAY_MS = max(0, int(os.getenv("OPENAI_INTER_CALL_DELAY_MS", "400")))

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
    "doctor_name",
    "doctor_registration_number",
    "doctor_signature",
    "diagnosis",
    "complaints",
    "prescribed_medicines",
    "advised_tests",
    "treatment_plan",
    "followup_date",
)

INVOICE_PARAM_KEYS: Tuple[str, ...] = (
    "patient_name",
    "invoice_number",
    "invoice_date",
    "provider_name",
    "provider_address",
    "provider_contact",
    "doctor_name",
    "service_details",
    "medicine_details",
    "test_details",
    "quantity",
    "unit_price",
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
)

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

_INVOICE_ARRAY_KEYS = frozenset({"service_details", "medicine_details", "test_details"})
_REPORT_ARRAY_KEYS = frozenset({"test_names", "test_results", "reference_ranges"})

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
    "invoice_number",
    "invoice_date",
    "doctor_name",
    "total_amount",
    "authorized_signature",
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
            props[key] = _STRING_ARRAY if key == "advised_tests" else {"type": "string"}
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

document_category: prescription | invoice | report. Rx without rupee total → prescription, NEVER invoice.

invoice_subtype (invoice only): pharmacy | diagnostic | opd_consultation | uncertain | not_applicable.

Invoice: test_details as "Test — Rs amount"; do not use printed row labels alone. Extract patient_name, doctor_name from handwriting.
Prescription: all medicines as {medicine,dosage}; advised_tests for labs only.

Report (lab/radiology):
- Formal lab report with TYPED/PRINTED patient, date, test name, numeric result, reference range → document_type=computer_generated, content_computer_generated_percent 95-100 (only signature may be handwritten ~0-10%).
- Do NOT mark printed lab reports as handwritten just because letterhead is printed.
- test_names: specific test (e.g. "Sr. Uric Acid"), not section title alone ("EXAMINATION OF BLOOD").
- pathologist_signature: "present" if handwritten signature visible.
"""

PRESCRIPTION_MEDICINE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "advised_tests": _STRING_ARRAY,
        "prescribed_medicines": {"type": "array", "items": _MEDICINE_ITEM_SCHEMA},
    },
    "required": ["advised_tests", "prescribed_medicines"],
    "additionalProperties": False,
}

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
        "doctor_name": {"type": "string"},
        "invoice_number": {"type": "string"},
        "invoice_date": {"type": "string"},
        "total_amount": {"type": "string"},
        "test_details": _STRING_ARRAY,
        "medicine_details": _STRING_ARRAY,
        "service_details": _STRING_ARRAY,
        "authorized_signature": {"type": "string"},
    },
    "required": [
        "document_type",
        "content_handwritten_percent",
        "content_computer_generated_percent",
        "patient_name",
        "doctor_name",
        "invoice_number",
        "invoice_date",
        "total_amount",
        "test_details",
        "medicine_details",
        "service_details",
        "authorized_signature",
    ],
    "additionalProperties": False,
}

DIAGNOSTIC_INVOICE_PROMPT = """Medical bill / cash memo image.

CONTENT (filled fields only):
- document_type: handwritten if patient, doctor, tests, prices, total are handwritten; computer_generated if all typed/printed.
- content_handwritten_percent + content_computer_generated_percent = 100 (content only, ignore blank template).

Extract: patient_name, doctor_name, invoice_number, invoice_date, total_amount, authorized_signature.
test_details: every test with price — "Vit D-3 — Rs 600" (not printed labels like Blood Examination alone).
service_details: extra fees only; medicine_details: drugs if pharmacy."""

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

PRESCRIPTION_MEDICINE_PROMPT = """Extract ONLY from this prescription (Rx) image.

advised_tests — lab tests only (Vit D3, CBC, uric acid). NOT medicines.

prescribed_medicines — EVERY drug line {medicine, dosage}; scan full page; do not stop after two items."""

DOCUMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": ["handwritten", "computer_generated", "uncertain"],
        },
        "content_handwritten_percent": {"type": "number"},
        "content_computer_generated_percent": {"type": "number"},
        "document_category": {
            "type": "string",
            "enum": ["prescription", "invoice", "report"],
        },
        "invoice_subtype": {
            "type": "string",
            "enum": [
                "pharmacy",
                "diagnostic",
                "opd_consultation",
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
        "document_category",
        "invoice_subtype",
        "prescription_parameters",
        "invoice_parameters",
        "report_parameters",
    ],
    "additionalProperties": False,
}


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
        elif key in array_keys:
            result[key] = _list_val(data.get(key))
        else:
            result[key] = _str_val(data.get(key))
    return result


def _merge_string_field(target: Dict[str, Any], key: str, value: str) -> None:
    if _str_val(value) and not _str_val(target.get(key)):
        target[key] = _str_val(value)


def _is_filled(params: Dict[str, Any], key: str) -> bool:
    val = params.get(key)
    if key == "prescribed_medicines" and isinstance(val, list):
        return any(
            isinstance(item, dict) and _str_val(item.get("medicine")) for item in val
        )
    if isinstance(val, list):
        return len(val) > 0
    return bool(_str_val(val))


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


def _infer_invoice_subtype(params: Dict[str, Any], raw_subtype: str) -> str:
    if raw_subtype in ("pharmacy", "diagnostic", "opd_consultation"):
        return raw_subtype
    if _is_filled(params, "medicine_details"):
        return "pharmacy"
    if _is_filled(params, "test_details"):
        return "diagnostic"
    if _is_filled(params, "service_details") or _is_filled(params, "doctor_name"):
        return "opd_consultation"
    if _is_filled(params, "drug_license_number") or _is_filled(params, "gst_number"):
        return "pharmacy"
    return "uncertain"


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
    """Cash memo mislabelled as computer_generated → content is handwritten."""
    doc_type = str(data.get("document_type", "uncertain"))
    tests = _list_val(inv_params.get("test_details"))
    services = _list_val(inv_params.get("service_details"))
    generic_only = bool(services) and all(_is_generic_invoice_service_line(s) for s in services)
    hw = float(data.get("content_handwritten_percent", 0))

    if doc_type == "computer_generated" and hw < 15 and (generic_only or tests):
        data["document_type"] = "handwritten"
        data["content_handwritten_percent"] = 100.0
        data["content_computer_generated_percent"] = 0.0


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


def _build_public_response(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    category = str(data.get("document_category", "report"))
    if category not in ("prescription", "invoice", "report"):
        category = "report"

    inv_params = _normalize_params(
        data.get("invoice_parameters"), INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS
    )
    if category == "invoice":
        _fix_content_classification(data, inv_params)
        inv_params = _normalize_params(
            data.get("invoice_parameters"), INVOICE_PARAM_KEYS, _INVOICE_ARRAY_KEYS
        )

    doc_type = _apply_document_type(data)
    content_hw, content_cg = _content_percent_split(data)
    category = _correct_document_category(doc_type, category, inv_params)

    if category == "prescription":
        parameters = _normalize_params(
            data.get("prescription_parameters"),
            PRESCRIPTION_PARAM_KEYS,
            frozenset({"advised_tests"}),
        )
        completeness, missing_parameters = _completeness(
            parameters,
            PRESCRIPTION_REQUIRED,
            (
                (
                    "diagnosis_or_complaints",
                    _is_filled(parameters, "diagnosis")
                    or _is_filled(parameters, "complaints"),
                ),
                (
                    "prescribed_medicines_or_advised_tests",
                    _is_filled(parameters, "prescribed_medicines")
                    or _is_filled(parameters, "advised_tests"),
                ),
            ),
        )
        invoice_subtype = "not_applicable"
    elif category == "invoice":
        parameters = inv_params
        completeness, missing_parameters = _completeness(
            parameters,
            INVOICE_REQUIRED,
            (
                (
                    "line_items",
                    _is_filled(parameters, "medicine_details")
                    or _is_filled(parameters, "test_details")
                    or _is_filled(parameters, "service_details"),
                ),
            ),
        )
        invoice_subtype = _infer_invoice_subtype(
            parameters, str(data.get("invoice_subtype", "uncertain"))
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
        "document_type": doc_type,
        "document_category": category,
        "invoice_subtype": invoice_subtype,
        "handwritten_percent": content_hw,
        "computer_generated_percent": content_cg,
        "completeness_percent": completeness,
        "parameters": parameters,
        "missing_parameters": missing_parameters,
    }


def _peek_prescription_category(data: Dict[str, Any]) -> str:
    doc_type = str(data.get("document_type", "uncertain"))
    category = str(data.get("document_category", "report"))
    inv_raw = data.get("invoice_parameters")
    inv = inv_raw if isinstance(inv_raw, dict) else {}
    total = _str_val(inv.get("total_amount"))
    if category == "invoice" and not total:
        if doc_type == "handwritten":
            return "prescription"
        if doc_type == "computer_generated" and not (
            _list_val(inv.get("test_details")) or _list_val(inv.get("medicine_details"))
        ):
            return "report"
    if category in ("prescription", "invoice", "report"):
        return category
    return "report"


def _merge_prescription_medicines(data: Dict[str, Any], refined: Dict[str, Any]) -> None:
    rx_raw = data.get("prescription_parameters")
    rx: Dict[str, Any] = dict(rx_raw) if isinstance(rx_raw, dict) else {}
    refined_meds = _normalize_prescribed_medicines(refined.get("prescribed_medicines"))
    refined_tests = _list_val(refined.get("advised_tests"))
    if refined_meds:
        rx["prescribed_medicines"] = refined_meds
    if refined_tests:
        rx["advised_tests"] = refined_tests
    data["prescription_parameters"] = rx


def _merge_diagnostic_invoice(data: Dict[str, Any], refined: Dict[str, Any]) -> None:
    inv_raw = data.get("invoice_parameters")
    inv: Dict[str, Any] = dict(inv_raw) if isinstance(inv_raw, dict) else {}

    for key in _INVOICE_REFINE_STRING_KEYS:
        _merge_string_field(inv, key, _str_val(refined.get(key)))

    tests = _list_val(refined.get("test_details"))
    medicines = _list_val(refined.get("medicine_details"))
    services = [
        s
        for s in _list_val(refined.get("service_details"))
        if not _is_generic_invoice_service_line(s)
    ]
    if tests:
        inv["test_details"] = tests
    if medicines:
        inv["medicine_details"] = medicines
    if services:
        inv["service_details"] = services
    elif tests:
        inv["service_details"] = [
            s
            for s in _list_val(inv.get("service_details"))
            if not _is_generic_invoice_service_line(s)
        ]

    data["invoice_parameters"] = inv

    if _str_val(refined.get("document_type")):
        data["document_type"] = refined["document_type"]
    if refined.get("content_handwritten_percent") is not None:
        data["content_handwritten_percent"] = refined["content_handwritten_percent"]
    if refined.get("content_computer_generated_percent") is not None:
        data["content_computer_generated_percent"] = refined["content_computer_generated_percent"]


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
    if str(data.get("document_category")) != "invoice":
        return False
    inv_raw = data.get("invoice_parameters")
    inv = inv_raw if isinstance(inv_raw, dict) else {}
    if not _str_val(inv.get("patient_name")):
        return True
    tests = _list_val(inv.get("test_details"))
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
            + " Fill only the matching parameter object."
        ),
        image_blocks,
        "medical_document_extraction",
        DOCUMENT_SCHEMA,
        3000,
    )


def _refine_prescription_medicines(
    client: Any,
    model: str,
    image_blocks: List[Dict[str, Any]],
    data: Dict[str, Any],
) -> None:
    rx_model = (os.getenv("OPENAI_RX_MODEL") or model).strip()
    refined = _call_openai_json(
        client,
        rx_model,
        PRESCRIPTION_MEDICINE_PROMPT,
        "List every advised test and every prescribed medicine on this Rx.",
        image_blocks,
        "prescription_medicine_extraction",
        PRESCRIPTION_MEDICINE_SCHEMA,
        2000,
    )
    _merge_prescription_medicines(data, refined)


def _refine_diagnostic_invoice(
    client: Any,
    model: str,
    image_blocks: List[Dict[str, Any]],
    data: Dict[str, Any],
) -> None:
    inv_model = (os.getenv("OPENAI_INVOICE_MODEL") or model).strip()
    refined = _call_openai_json(
        client,
        inv_model,
        DIAGNOSTIC_INVOICE_PROMPT,
        "Extract content type, patient, doctor, and all test lines with prices.",
        image_blocks,
        "diagnostic_invoice_extraction",
        DIAGNOSTIC_INVOICE_SCHEMA,
        2000,
    )
    _merge_diagnostic_invoice(data, refined)


def _refine_lab_report(
    client: Any,
    model: str,
    image_blocks: List[Dict[str, Any]],
    data: Dict[str, Any],
) -> None:
    report_model = (os.getenv("OPENAI_REPORT_MODEL") or model).strip()
    refined = _call_openai_json(
        client,
        report_model,
        LAB_REPORT_PROMPT,
        "Classify printed vs handwritten content and extract lab test fields.",
        image_blocks,
        "lab_report_extraction",
        LAB_REPORT_SCHEMA,
        2000,
    )
    _merge_lab_report(data, refined)


def classify_document_url_openai(url: str) -> Dict[str, Any]:
    """Classify one image URL and return category-specific extracted parameters."""
    model = (os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    client = get_openai_client()
    image_blocks, _ = build_vision_image_blocks(url)
    data = _call_openai_vision(client, model, image_blocks)
    if _peek_prescription_category(data) == "prescription":
        try:
            _pause_between_openai_calls()
            _refine_prescription_medicines(client, model, image_blocks, data)
        except Exception:
            logger.exception("Prescription medicine refine pass failed for %s", url)
    elif _peek_invoice_needs_refine(data):
        try:
            _pause_between_openai_calls()
            _refine_diagnostic_invoice(client, model, image_blocks, data)
        except Exception:
            logger.exception("Diagnostic invoice refine pass failed for %s", url)
    elif _peek_report_needs_refine(data):
        try:
            _pause_between_openai_calls()
            _refine_lab_report(client, model, image_blocks, data)
        except Exception:
            logger.exception("Lab report refine pass failed for %s", url)
    return _build_public_response(url, data)
