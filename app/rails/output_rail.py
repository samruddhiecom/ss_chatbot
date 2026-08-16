"""Output rail: the grounding / claims-clean gate.

Two mechanisms, cheapest first:
  1. no_invented_numbers(): deterministic, zero model calls. Flags statistic-like
     figures in the output that do not appear in the sources. This is the ported
     verbatim-grounding discipline pointed at numbers, which are the highest-risk
     fabrication in a planning snapshot.
  2. claims_clean(): one light LLM self-check for synthesised prose, listing any
     claim not supported by the sources.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from app import llm
from app.schemas import GroundingReport

# Statistic-like figures: percentages, multi-digit numbers, currency amounts, and
# multipliers like "10x". Single small integers (1-9 on their own) are ignored so
# that "4 stages" or "three channels" do not trip the guard.
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%|\d[\d,]*(?:\.\d+)?\s?[xX]\b|\d{2,}(?:[.,]\d+)?)"
)


def _normalise(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower())


def no_invented_numbers(text: str, sources: str) -> List[str]:
    """Return statistic-like figures present in `text` but absent from `sources`."""
    src = _normalise(sources)
    offenders: List[str] = []
    for m in _NUMBER_RE.finditer(text or ""):
        token = m.group(0)
        digits = _normalise(token)
        # Keep only the digit core for comparison (strip $, %, x).
        core = re.sub(r"[^\d.]", "", digits)
        if core and core not in src:
            offenders.append(token.strip())
    return offenders


def claims_clean(text: str, sources: str) -> GroundingReport:
    """LLM self-check. Falls back to 'supported' if no model is configured, since the
    deterministic guard has already run and this is a secondary gate.
    """
    if not llm.settings.has_llm:
        return GroundingReport(supported=True, unsupported_claims=[])
    system = (
        "You verify grounding. Given SOURCES and TEXT, list any factual claim in TEXT that is "
        "not supported by SOURCES. Organising or rephrasing the founder's own words counts as "
        "supported. Inventing facts, figures, or market data counts as unsupported. "
        "If everything is supported, return supported=true with an empty list."
    )
    user = f"SOURCES:\n{sources}\n\nTEXT:\n{text}"
    return llm.structured(system, user, GroundingReport, temperature=0.0)


def verify_turn(text: str, sources: str) -> Tuple[bool, List[str]]:
    """Fast gate used on each assistant turn. Returns (clean, flags)."""
    flags = no_invented_numbers(text, sources)
    return (len(flags) == 0, flags)
