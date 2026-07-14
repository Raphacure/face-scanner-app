from typing import List

from pydantic import BaseModel, Field

from fastapi import APIRouter

from app.controllers.receipt_controller import (
    MAX_URLS,
    classify_receipt_prescription_urls_controller,
)

router = APIRouter(prefix="/documents", tags=["documents"])


class ReceiptPrescriptionClassifyRequest(BaseModel):
    """Public image/PDF URLs to classify (Textract OCR + OpenAI Vision hybrid)."""

    urls: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_URLS,
        description="Publicly reachable image or PDF URLs.",
    )


@router.post("/classify-receipt")
def classify_receipt_prescription(payload: ReceiptPrescriptionClassifyRequest):
    """Hybrid classify: Textract (printed fields) + OpenAI Vision (category/medical fields)."""
    return classify_receipt_prescription_urls_controller(payload.urls)
