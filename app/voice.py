"""Layer C brand voice, applied AFTER grounded content is produced.

Sequencing, not disentanglement: since style and content cannot be cleanly separated
inside one sentence, we separate them in the pipeline instead. Content is locked first;
voice restyles the surface; then a content-preservation check confirms the restyle did
not introduce a figure that was not there.
"""
from __future__ import annotations

from functools import lru_cache

from config import DRAFT_KB_DIR
from app import llm


@lru_cache(maxsize=1)
def voice_spec() -> str:
    return (DRAFT_KB_DIR / "layer_c_voice.md").read_text(encoding="utf-8")


def apply_voice(text: str, *, sources: str = "") -> str:
    """Restyle `text` per the SS voice without changing any fact.

    If no model is configured, return the text unchanged. If the restyle introduces a
    statistic-like number that was not in the source text, revert (grounding wins).
    """
    if not text or not llm.settings.has_llm:
        return text

    system = (
        "Rewrite the assistant message in Simplified Startup's brand voice. "
        "Change wording and rhythm only. Do NOT add, remove, or alter any fact, number, "
        "name, or claim. Keep it short.\n\n"
        f"{voice_spec()}"
    )
    styled = llm.generate(system, text, temperature=0.3)

    # Content preservation: the restyle must not introduce new figures.
    from app.rails import output_rail

    baseline = f"{text}\n{sources}"
    if output_rail.no_invented_numbers(styled, baseline):
        return text  # revert to the grounded original
    return styled
