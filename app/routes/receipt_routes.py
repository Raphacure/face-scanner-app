from typing import List

from pydantic import BaseModel, Field

from fastapi import APIRouter

from app.controllers.receipt_controller import (
    MAX_URLS,
    classify_receipt_prescription_urls_controller,
)

router = APIRouter(prefix="/documents", tags=["documents"])


class ReceiptPrescriptionClassifyRequest(BaseModel):
    """Public image URLs of medical documents (JPEG/PNG/WebP or PDF) to classify via OpenAI Vision."""

    urls: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_URLS,
        description="Publicly reachable image or PDF URLs.",
    )


@router.post("/classify-receipt")
def classify_receipt_prescription(payload: ReceiptPrescriptionClassifyRequest):
    """
    For each image URL: document_category (prescription/invoice/report/other),
    extracted parameters, or rejection when the image is not a medical document.
    Requires OPENAI_API_KEY.
    """
    return classify_receipt_prescription_urls_controller(payload.urls)
