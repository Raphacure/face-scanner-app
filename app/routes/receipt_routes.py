from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.controllers.receipt_controller import (
    MAX_URLS,
    classify_receipt_prescription_urls_controller,
    get_classify_job_controller,
    submit_classify_job_controller,
)
from app.services.openai_document_classifier import (
    normalize_document_name_hint,
    normalize_extract_fields,
)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentClassifyItem(BaseModel):
    """One document URL with CRM category name."""

    url: str = Field(..., min_length=1, description="Publicly reachable image or PDF URL.")
    name: str = Field(
        ...,
        min_length=1,
        description="CRM document label (invoice, prescription, report, payment_receipt, other, etc.).",
    )
    fields: Optional[List[str]] = Field(
        default=None,
        description=(
            "Field names to extract from this document. "
            "The OpenAI schema is built from this list. "
            "Omit to extract the full default claim catalog."
        ),
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not text:
            raise ValueError("name is required")
        # Map known aliases; pass through anything else (e.g. other) — CRM validates.
        mapped = normalize_document_name_hint(value)
        return mapped if mapped else text

    @field_validator("fields")
    @classmethod
    def _validate_fields(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        normalized = normalize_extract_fields(value)
        if not normalized:
            raise ValueError(
                "fields must be a non-empty list of field names "
                "(letters, numbers, underscore)"
            )
        return list(normalized)


class ReceiptPrescriptionClassifyRequest(BaseModel):
    """Classify claim documents (Textract OCR + OpenAI Vision hybrid)."""

    model_config = ConfigDict(populate_by_name=True)

    urls: List[DocumentClassifyItem] = Field(
        ...,
        min_length=1,
        max_length=MAX_URLS,
        description='Array of {"url", "name", optional "fields"} objects.',
    )
    async_job: bool = Field(
        default=False,
        alias="async",
        description=(
            "If true, return immediately with job_id — poll GET "
            "/classify-receipt/jobs/{job_id} (avoids gateway timeout)."
        ),
    )


@router.post("/classify-receipt")
async def classify_receipt_prescription(payload: ReceiptPrescriptionClassifyRequest):
    """Extract fields using CRM-provided document name.

    Sync (default): waits and returns results (may hit 60s gateway timeout).
    Async (`"async": true`): returns job_id in <1s; poll the job URL for results.
    """
    if payload.async_job:
        body = submit_classify_job_controller(payload.urls)
        code = 202 if body.get("status") == "accepted" else 400
        return JSONResponse(status_code=code, content=body)
    return classify_receipt_prescription_urls_controller(payload.urls)


@router.get("/classify-receipt/jobs/{job_id}")
async def get_classify_receipt_job(job_id: str):
    """Poll async classify job until status is success or error."""
    body = get_classify_job_controller(job_id)
    if body.get("status") in ("pending", "processing"):
        return JSONResponse(status_code=202, content=body)
    if body.get("status") == "error" and body.get("message", "").startswith("job not found"):
        return JSONResponse(status_code=404, content=body)
    return body
