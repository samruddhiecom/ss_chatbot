"""Retrieval rail: the only path by which external facts reach generation.

Generation is never allowed to cite anything outside what this returns, which is
the operational form of the allowlist pattern.
"""
from __future__ import annotations

from typing import List

from app.kb import store


def for_stage(stage: str, query_text: str) -> List[str]:
    """Layer-A guidance for this stage + stage-agnostic Layer-B facts."""
    return store.query(query_text, stage=stage)


def all_grounding() -> List[str]:
    """Whole allowlist, for snapshot-time grounding."""
    return store.query_all_grounding()
