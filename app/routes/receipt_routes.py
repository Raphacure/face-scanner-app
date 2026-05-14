from typing import List

from pydantic import BaseModel, Field

from fastapi import APIRouter

from app.controllers.receipt_controller import classify_receipt_prescription_urls_controller

router = APIRouter(prefix="/documents", tags=["documents"])


class ReceiptPrescriptionClassifyRequest(BaseModel):
    """Image URLs of documents to separate receipts from handwritten prescriptions."""

    urls: List[str] = Field(
        ...,
        min_length=1,
        description="Publicly reachable image URLs (e.g. PNG/JPEG).",
    )


@router.post("/classify-receipt")
def classify_receipt_prescription(payload: ReceiptPrescriptionClassifyRequest):
    """
    For each image URL: `document_type` plus `handwritten_percent` and
    `computer_generated_percent` (heuristic split; they sum to 100).
    """
    return classify_receipt_prescription_urls_controller(payload.urls)
