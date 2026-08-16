"""Deterministic grounding checks. No model, no network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rails import output_rail


def test_flags_invented_percentage():
    text = "Your market is growing at 47% a year."
    assert output_rail.no_invented_numbers(text, sources="the founder said nothing about growth")


def test_flags_invented_market_size():
    text = "The total market is worth $12,000,000."
    assert output_rail.no_invented_numbers(text, sources="a small local bakery idea")


def test_allows_grounded_number():
    text = "A first version typically lands in about 4 to 6 weeks."
    sources = "a first live, revenue-ready version typically lands in about four to six weeks (4-6)"
    # 4 and 6 are single digits -> not treated as statistic-like, so no flag either way.
    assert output_rail.no_invented_numbers(text, sources) == []


def test_allows_number_present_in_sources():
    text = "You mentioned 150 sign-ups already."
    sources = "Founder: we have 150 sign-ups on the waitlist"
    assert output_rail.no_invented_numbers(text, sources) == []


def test_ignores_small_counts():
    text = "There are 4 stages and 3 channels to consider."
    assert output_rail.no_invented_numbers(text, sources="") == []


def test_verify_turn_returns_flags():
    clean, flags = output_rail.verify_turn("Growth is 30% monthly.", sources="")
    assert clean is False
    assert flags
