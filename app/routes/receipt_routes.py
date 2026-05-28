from typing import List

from pydantic import BaseModel, Field

from fastapi import APIRouter

from app.controllers.receipt_controller import (
    MAX_URLS,
    classify_receipt_prescription_urls_controller,
)

router = APIRouter(prefix="/documents", tags=["documents"])


class ReceiptPrescriptionClassifyRequest(BaseModel):
    """Image URLs of documents to separate receipts from handwritten prescriptions."""

    urls: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_URLS,
        description="Publicly reachable image URLs (e.g. PNG/JPEG).",
    )


@router.post("/classify-receipt")
def classify_receipt_prescription(payload: ReceiptPrescriptionClassifyRequest):
    """
    For each image URL: primary document_type
    (computer_generated/handwritten/uncertain), plus document_category
    (prescription/invoice/report), and prescription completeness
    (stamp, hospital name/address, doctor name/signature, consultation type,
    amount). Each detected field adds ~14.29% to completeness_percent.
    """
    return classify_receipt_prescription_urls_controller(payload.urls)
