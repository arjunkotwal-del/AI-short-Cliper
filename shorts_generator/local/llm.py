"""Local LLM backend — calls an OpenAI-compatible Chat Completions API directly.

Works with OpenAI itself, or any OpenAI-compatible provider through OPENAI_BASE_URL
(e.g. Google Gemini: https://generativelanguage.googleapis.com/v1beta/openai/).
The openai client reads OPENAI_BASE_URL from the environment on its own.
"""
from ..config import OPENAI_MODEL, require_openai_key

_TIMEOUT = 120  # seconds per request
_MAX_RETRIES = 2


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend with timeout and retry."""
    import os
    if os.getenv("MOCK_LLM", "false").lower() == "true":
        from .mock_llm import mock_response
        return mock_response(prompt)

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from e

    client = OpenAI(
        api_key=require_openai_key(),
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""
