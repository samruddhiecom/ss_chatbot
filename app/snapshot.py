"""Assemble the durable Snapshot from the transcript + allowlist, then verify it.

The snapshot organises only what the founder said plus reviewed KB facts. Every gap
the conversation surfaced (deferrals, empty stages) becomes a named gap with a next
step, which is how refusals become discovery value.
"""
from __future__ import annotations

from typing import List

from app import llm, voice
from app.rails import output_rail, retrieval_rail
from app.schemas import GroundingReport, Snapshot

_SYSTEM = """You assemble a startup planning snapshot for Simplified Startup.

Hard rules:
- Organise ONLY what the founder said and the provided KB facts. Invent nothing.
- No market statistics, no revenue projections, no valuations, no professional advice.
- If something was not covered, do not fill it in; record it under named_gaps.
- channel_shortlist may only contain channels the founder named or that follow directly.
- next_steps should be concrete and, where possible, one per gap (prefer quick validation actions).
Return the structured snapshot."""


def assemble(messages: List[dict], carried_gaps: List[str]) -> tuple[Snapshot, GroundingReport]:
    transcript = llm.transcript_text(messages)
    kb = "\n".join(retrieval_rail.all_grounding())
    sources = f"KNOWLEDGE BASE:\n{kb}\n\nTRANSCRIPT:\n{transcript}"

    if not llm.settings.has_llm:
        # Degraded, model-free fallback so the endpoint still returns a shape.
        snap = Snapshot(
            idea_framing="(Model not configured — set GROQ_API_KEY to generate the snapshot.)",
            customer_hypothesis="",
            offer_sketch="",
            channel_shortlist=[],
            named_gaps=carried_gaps or ["Snapshot generation requires a configured model."],
            next_steps=["Configure GROQ_API_KEY and re-run."],
        )
        return snap, GroundingReport(supported=True, unsupported_claims=[])

    snap: Snapshot = llm.structured(
        system=_SYSTEM,
        user=sources,
        response_model=Snapshot,
        temperature=0.2,
    )

    # Merge deferral-driven gaps the conversation already surfaced (dedup, order-stable).
    merged: List[str] = list(snap.named_gaps)
    for g in carried_gaps:
        if g and g not in merged:
            merged.append(g)
    snap.named_gaps = merged

    # Voice pass on the prose fields, fact-preserving.
    snap.idea_framing = voice.apply_voice(snap.idea_framing, sources=sources)
    snap.customer_hypothesis = voice.apply_voice(snap.customer_hypothesis, sources=sources)
    snap.offer_sketch = voice.apply_voice(snap.offer_sketch, sources=sources)

    # Claims-clean gate over the synthesised prose.
    prose = "\n".join([snap.idea_framing, snap.customer_hypothesis, snap.offer_sketch])
    report = output_rail.claims_clean(prose, sources)
    return snap, report
