"""
Classify medical document images with OpenAI Vision + structured JSON output.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from app.core.openai_client import get_openai_client
from app.services.document_image_fetch import download_image_data_url

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FIELD_KEYS = (
    "stamp",
    "hospital_name",
    "hospital_address",
    "doctor_name",
    "doctor_signature",
    "consultation_type",
    "amount",
)

SYSTEM_PROMPT = """You are a medical document analyst for Indian healthcare paperwork.

Classify the image accurately:

document_type (how the transaction details were filled — not just the blank form):
- handwritten: Rx notes, OR pharmacy/chemist bill where patient name, medicines, qty, rates,
  totals are written by hand (even if the form template/letterhead is pre-printed)
- computer_generated: fully printed/digital bill, receipt, lab report, audiogram (no handwritten
  line items or totals)
- uncertain: cannot tell

document_category (what kind of document):
- prescription: doctor Rx / medicines / dosage advice; NOT a paid pharmacy cash memo
- invoice: bill/receipt with amounts and line items (hospital OP bill, pharmacy bill, tax invoice)
- report: lab/diagnostic/clinical report (audiogram, blood test) without billing as the focus

For each completeness field, set likely_present true only if clearly visible in the image.
confidence_percent is 0-100 (how sure you are that the field is present).

Rules:
- Printed hospital letterhead on a handwritten Rx pad → handwritten + prescription
- Pharmacy/medical store bill pad: printed headers but handwritten table rows and total →
  handwritten + invoice (NOT computer_generated)
- Hospital OP bill / bill cum receipt fully printed → computer_generated + invoice
- Audiogram, pure tone test, diagnostic tables without billing → computer_generated + report
- Stamp means rubber stamp/seal, not just printed logo
- amount means billing/payment amounts (Rs/INR/total), not clinical measurements (dB, Hz)
- handwritten_percent should be high when handwritten content dominates; computer_generated_percent
  high only for fully printed/digital documents
"""

DOCUMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": ["handwritten", "computer_generated", "uncertain"],
        },
        "document_category": {
            "type": "string",
            "enum": ["prescription", "invoice", "report"],
        },
        "handwritten_percent": {"type": "number"},
        "computer_generated_percent": {"type": "number"},
        "fields": {
            "type": "object",
            "properties": {
                key: {
                    "type": "object",
                    "properties": {
                        "likely_present": {"type": "boolean"},
                        "confidence_percent": {"type": "number"},
                    },
                    "required": ["likely_present", "confidence_percent"],
                    "additionalProperties": False,
                }
                for key in FIELD_KEYS
            },
            "required": list(FIELD_KEYS),
            "additionalProperties": False,
        },
    },
    "required": [
        "document_type",
        "document_category",
        "handwritten_percent",
        "computer_generated_percent",
        "fields",
    ],
    "additionalProperties": False,
}


def _normalize_field_entry(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"likely_present": False, "confidence_percent": 0.0}
    present = bool(raw.get("likely_present"))
    conf = float(raw.get("confidence_percent", 0.0))
    conf = max(0.0, min(100.0, conf))
    return {
        "likely_present": present,
        "confidence_percent": round(conf, 2),
    }


def _build_public_response(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    doc_type = str(data.get("document_type", "uncertain"))
    if doc_type not in ("handwritten", "computer_generated", "uncertain"):
        doc_type = "uncertain"

    category = str(data.get("document_category", "report"))
    if category not in ("prescription", "invoice", "report"):
        category = "report"

    hw = max(0.0, min(100.0, float(data.get("handwritten_percent", 50.0))))
    cg = max(0.0, min(100.0, float(data.get("computer_generated_percent", 50.0))))
    total = hw + cg
    if total < 1e-6:
        hw, cg = 50.0, 50.0
    else:
        hw = round(hw / total * 100.0, 2)
        cg = round(100.0 - hw, 2)

    raw_fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    fields: Dict[str, Any] = {}
    present_count = 0
    weight = 100.0 / len(FIELD_KEYS)
    completeness = 0.0

    for key in FIELD_KEYS:
        entry = _normalize_field_entry(raw_fields.get(key))
        if entry["likely_present"]:
            present_count += 1
            completeness += weight
            entry["contribution_percent"] = round(weight, 2)
        else:
            entry["contribution_percent"] = 0.0
        fields[key] = entry

    return {
        "url": url,
        "document_type": doc_type,
        "document_category": category,
        "handwritten_percent": hw,
        "computer_generated_percent": cg,
        "completeness_percent": round(completeness, 2),
        "present_count": present_count,
        "total_fields": len(FIELD_KEYS),
        "fields": fields,
        "classification_source": "openai",
    }


def _image_content_from_url(url: str) -> Dict[str, Any]:
    """Download image and send as base64 (OpenAI often cannot fetch private/slow S3)."""
    data_url, _ = download_image_data_url(url)
    return {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}


def _call_openai_vision(
    client: Any,
    model: str,
    user_text: str,
    image_content: Dict[str, Any],
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    image_content,
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "medical_document_classification",
                "strict": True,
                "schema": DOCUMENT_SCHEMA,
            },
        },
        temperature=0.1,
        max_tokens=1200,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from OpenAI")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("OpenAI response is not a JSON object")
    return data


def classify_document_url_openai(url: str) -> Dict[str, Any]:
    """Classify one public image URL; returns API-ready result dict."""
    model = (os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    client = get_openai_client()

    user_text = (
        "Analyze this medical document image and return JSON matching the schema. "
        "Pharmacy bill pads with handwritten patient/items/totals → handwritten + invoice. "
        "Fully printed hospital bills → computer_generated + invoice."
    )

    data = _call_openai_vision(
        client, model, user_text, _image_content_from_url(url)
    )
    return _build_public_response(url, data)
