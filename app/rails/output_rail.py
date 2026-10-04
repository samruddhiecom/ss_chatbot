"""Output rail: the grounding / claims-clean gate.

Two mechanisms, cheapest first:
  1. no_invented_numbers(): deterministic, zero model calls. Flags statistic-like
     figures in the output that do not appear in the sources.
  2. brand_name_check(): deterministic, zero model calls. Flags "SS" abbreviation
     which is a hard KB guardrail violation.
  3. result_promise_check(): deterministic, zero model calls. Flags result-promise
     language that the KB guardrails explicitly ban.
  4. regenerate_if_flagged(): one regeneration pass with a tighter prompt when
     any flag fires. Ships the regenerated reply even if it still has issues,
     since one retry is the maximum.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from app import llm
from app.schemas import GroundingReport

# Statistic-like figures: percentages, multi-digit numbers, currency amounts,
# and multipliers like "10x". Single small integers (1-9) are ignored.
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%|\d[\d,]*(?:\.\d+)?\s?[xX]\b|\d{2,}(?:[.,]\d+)?)"
)

# Matches "SS" as a standalone abbreviation — word boundary on both sides.
_SS_ABBREV_RE = re.compile(r"\bSS\b")

# Result-promise language banned by the KB guardrails.
_RESULT_PROMISE_RE = re.compile(
    r"\b(we('ll| will) (get you|drive|grow|generate|deliver|guarantee|produce|increase)|"
    r"guaranteed? (results?|growth|leads?|sales?|revenue)|"
    r"(specializes?|built|designed|made) (to|for) (solve|fix|resolve|help)|"
    r"turn(s|ing)? (traffic|visitors?|leads?) into (sales?|revenue|customers?|buyers?)|"
    r"we (fix|solve|resolve|handle) (that|this|it)|"
    r"(proven|tested) (results?|system|method|approach))\b",
    re.IGNORECASE,
)


def _normalise(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower())


def no_invented_numbers(text: str, sources: str) -> List[str]:
    """Return statistic-like figures present in text but absent from sources."""
    src = _normalise(sources)
    offenders: List[str] = []
    for m in _NUMBER_RE.finditer(text or ""):
        token = m.group(0)
        digits = _normalise(token)
        core = re.sub(r"[^\d.]", "", digits)
        if core and core not in src:
            offenders.append(token.strip())
    return offenders


def brand_name_check(text: str) -> List[str]:
    """Flag use of 'SS' abbreviation — hard KB guardrail violation."""
    flags = []
    for m in _SS_ABBREV_RE.finditer(text or ""):
        flags.append(
            f"Brand name violation: '{m.group(0)}' used instead of 'Simplified Startup' at position {m.start()}"
        )
    return flags


def result_promise_check(text: str) -> List[str]:
    """Flag result-promise language banned by the KB guardrails."""
    flags = []
    for m in _RESULT_PROMISE_RE.finditer(text or ""):
        flags.append(f"Result-promise language: '{m.group(0)}'")
    return flags


def claims_clean(text: str, sources: str) -> GroundingReport:
    """LLM self-check. Falls back to 'supported' if no model is configured."""
    if not llm.settings.has_llm:
        return GroundingReport(supported=True, unsupported_claims=[])
    system = (
        "You verify grounding. Given SOURCES (a knowledge base plus a conversation transcript) and TEXT, "
        "list any factual claim in TEXT that is NOT supported by SOURCES. "
        "Important: claims that come from the founder's own statements in the TRANSCRIPT are considered supported -- "
        "they are the founder's words being organised, not invented facts. "
        "Only flag claims that are invented, fabricated, or not traceable to either the KB or the transcript. "
        "If everything is supported, return supported=true with an empty list."
    )
    user = f"SOURCES:\n{sources}\n\nTEXT:\n{text}"
    return llm.structured(system, user, GroundingReport, temperature=0.0)


def verify_turn(text: str, sources: str) -> Tuple[bool, List[str]]:
    """Fast deterministic gate used on each assistant turn. Returns (clean, flags)."""
    flags = no_invented_numbers(text, sources)
    flags += brand_name_check(text)
    flags += result_promise_check(text)
    return (len(flags) == 0, flags)


def regenerate_if_flagged(
    text: str,
    sources: str,
    regenerate_fn: Callable[[str], str],
) -> Tuple[str, List[str]]:
    """Run the output rail. If flags fire, regenerate once with a tighter prompt.

    Args:
        text:          The original reply from the advisor.
        sources:       KB chunks + conversation transcript for grounding check.
        regenerate_fn: A callable that takes a correction prompt and returns a
                       new reply string. Called at most once.

    Returns:
        (final_reply, flags_from_original)
    """
    clean, flags = verify_turn(text, sources)
    if clean or not llm.settings.has_llm:
        return text, flags

    # Build a correction prompt listing exactly what was wrong.
    issues = "\n".join(f"- {f}" for f in flags)
    correction_prompt = (
        f"Your previous reply was flagged for the following issues:\n{issues}\n\n"
        f"Previous reply:\n{text}\n\n"
        "Rewrite the reply fixing ALL of the above issues. "
        "Never use 'SS' -- always write 'Simplified Startup' in full. "
        "Never promise or imply results -- describe what Simplified Startup does, not what it will achieve. "
        "Never include figures that do not appear in the reference information. "
        "Keep the same tone and intent, just fix the violations."
    )

    try:
        new_reply = regenerate_fn(correction_prompt)
        return new_reply, flags
    except Exception:
        # If regeneration fails for any reason, ship the original rather than crashing.
        return text, flags