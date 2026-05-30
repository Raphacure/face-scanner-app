from typing import List

from pydantic import BaseModel, Field

from fastapi import APIRouter

from app.controllers.receipt_controller import (
    MAX_URLS,
    classify_receipt_prescription_urls_controller,
)

router = APIRouter(prefix="/documents", tags=["documents"])


class ReceiptPrescriptionClassifyRequest(BaseModel):
    """Public image URLs of medical documents to classify via OpenAI Vision."""

    urls: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_URLS,
        description="Publicly reachable image URLs (e.g. PNG/JPEG).",
    )


@router.post("/classify-receipt")
def classify_receipt_prescription(payload: ReceiptPrescriptionClassifyRequest):
    """
    For each image URL: document_type (computer_generated/handwritten/uncertain),
    document_category (prescription/invoice/report), completeness fields, and
    percentages. Classification uses OpenAI Vision only (requires OPENAI_API_KEY).
    """
    return classify_receipt_prescription_urls_controller(payload.urls)
