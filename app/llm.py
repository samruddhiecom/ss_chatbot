"""Groq access, wrapped so the rest of the app never touches the SDK directly.

Two call shapes:
  - generate(): plain text completion (assistant turns, refusals, voice pass)
  - structured(): Instructor-parsed Pydantic output (intent, extraction, snapshot, grounding)

The client is created lazily so the app (and KB ingestion, which needs no LLM)
imports fine without a key. Missing-key failures surface only when a live call is made.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Type, TypeVar

from pydantic import BaseModel

from config import settings

T = TypeVar("T", bound=BaseModel)


class LLMNotConfigured(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _raw_client():
    if not settings.has_llm:
        raise LLMNotConfigured(
            "GROQ_API_KEY is not set. Add it to .env before making live model calls."
        )
    from groq import Groq

    return Groq(api_key=settings.groq_api_key)


@lru_cache(maxsize=1)
def _instructor_client():
    import instructor

    return instructor.from_groq(_raw_client(), mode=instructor.Mode.JSON)


def generate(system: str, user: str, *, model: str | None = None, temperature: float | None = None) -> str:
    """Plain text generation."""
    client = _raw_client()
    resp = client.chat.completions.create(
        model=model or settings.smart_model,
        temperature=settings.temperature if temperature is None else temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def structured(system: str, user: str, response_model: Type[T], *, model: str | None = None,
               temperature: float | None = None) -> T:
    """Instructor-parsed structured output into `response_model`."""
    client = _instructor_client()
    return client.chat.completions.create(
        model=model or settings.smart_model,
        temperature=settings.temperature if temperature is None else temperature,
        response_model=response_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_retries=2,
    )


def transcript_text(messages: List[dict]) -> str:
    """Flatten a message list into a readable transcript for grounding sources."""
    lines = []
    for m in messages:
        who = "Founder" if m.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {m.get('content', '')}")
    return "\n".join(lines)
