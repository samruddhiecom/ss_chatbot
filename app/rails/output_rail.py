"""Output rail: the grounding / claims-clean gate.

Two mechanisms, cheapest first:
  1. no_invented_numbers(): deterministic, zero model calls. Flags statistic-like
     figures in the output that do not appear in the sources.
  2. brand_name_check(): deterministic, zero model calls. Flags "SS" abbreviation
     which is a hard KB guardrail violation.
  3. claims_clean(): one light LLM self-check for synthesised prose.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from app import llm
from app.schemas import GroundingReport

# Statistic-like figures: percentages, multi-digit numbers, currency amounts,
# and multipliers like "10x". Single small integers (1-9) are ignored.
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%|\d[\d,]*(?:\.\d+)?\s?[xX]\b|\d{2,}(?:[.,]\d+)?)"
)

# Matches "SS" as a standalone abbreviation — word boundary on both sides,
# not part of a longer word. Case-sensitive: "SS" only, not "ss".
_SS_ABBREV_RE = re.compile(r"\bSS\b")


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
    """Flag use of 'SS' abbreviation — hard KB guardrail violation.
    The bot must always say 'Simplified Startup' in full."""
    flags = []
    for m in _SS_ABBREV_RE.finditer(text or ""):
        flags.append(f"Brand name violation: '{m.group(0)}' used instead of 'Simplified Startup' at position {m.start()}")
    return flags


def claims_clean(text: str, sources: str) -> GroundingReport:
    """LLM self-check. Falls back to 'supported' if no model is configured."""
    if not llm.settings.has_llm:
        return GroundingReport(supported=True, unsupported_claims=[])
    system = (
        "You verify grounding. Given SOURCES (a knowledge base plus a conversation transcript) and TEXT, "
        "list any factual claim in TEXT that is NOT supported by SOURCES. "
        "Important: claims that come from the founder's own statements in the TRANSCRIPT are considered supported — "
        "they are the founder's words being organised, not invented facts. "
        "Only flag claims that are invented, fabricated, or not traceable to either the KB or the transcript. "
        "If everything is supported, return supported=true with an empty list."
    )
    user = f"SOURCES:\n{sources}\n\nTEXT:\n{text}"
    return llm.structured(system, user, GroundingReport, temperature=0.0)


def verify_turn(text: str, sources: str) -> Tuple[bool, List[str]]:
    """Fast gate used on each assistant turn. Returns (clean, flags)."""
    flags = no_invented_numbers(text, sources)
    flags += brand_name_check(text)
    return (len(flags) == 0, flags)