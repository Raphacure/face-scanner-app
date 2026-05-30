"""Shared OpenAI client (API key from environment only)."""

from __future__ import annotations

import os

from openai import OpenAI

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _client
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if _client is None:
        _client = OpenAI(api_key=api_key)
    return _client


def openai_configured() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())
