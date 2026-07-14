"""AWS Textract client for document OCR."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import boto3
from dotenv import load_dotenv

load_dotenv()

# Set after AccessDenied so we stop calling Textract for the process lifetime.
_textract_disabled_runtime = False


def disable_textract_runtime() -> None:
    global _textract_disabled_runtime
    _textract_disabled_runtime = True


def textract_enabled() -> bool:
    """Textract hybrid OCR — on by default when AWS credentials exist."""
    if _textract_disabled_runtime:
        return False
    flag = (os.getenv("USE_TEXTRACT") or "true").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return bool(
        (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
        and (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
    )


@lru_cache(maxsize=1)
def get_textract_client() -> Any:
    region = (os.getenv("AWS_REGION") or "ap-south-1").strip()
    return boto3.client(
        "textract",
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
        region_name=region,
    )
