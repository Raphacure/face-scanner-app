from typing import List

from pydantic import BaseModel, Field, field_validator

from fastapi import APIRouter

from app.controllers.receipt_controller import (
    MAX_URLS,
    classify_receipt_prescription_urls_controller,
)
from app.services.openai_document_classifier import normalize_document_name_hint

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentClassifyItem(BaseModel):
    """One document URL with CRM category name."""

    url: str = Field(..., min_length=1, description="Publicly reachable image or PDF URL.")
    name: str = Field(
        ...,
        min_length=1,
        description="invoice | prescription | report | payment_receipt",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = normalize_document_name_hint(value)
        if not normalized:
            raise ValueError(
                "name must be invoice, prescription, report, or payment_receipt "
                "(aliases: bill, rx, opd, lab, payment, upi)"
            )
        return normalized


class ReceiptPrescriptionClassifyRequest(BaseModel):
    """Classify claim documents (Textract OCR + OpenAI Vision hybrid)."""

    urls: List[DocumentClassifyItem] = Field(
        ...,
        min_length=1,
        max_length=MAX_URLS,
        description='Array of {"url", "name"} objects.',
    )


@router.post("/classify-receipt")
def classify_receipt_prescription(payload: ReceiptPrescriptionClassifyRequest):
    """Extract fields using CRM-provided document name (invoice / prescription / report / payment_receipt)."""
    return classify_receipt_prescription_urls_controller(payload.urls)
