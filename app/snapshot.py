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
- Organise ONLY what the founder said in the TRANSCRIPT. Invent nothing.
- The snapshot is about the FOUNDER'S idea, not about Simplified Startup.
- Never include SS company language, SS services, SS marketing copy, or SS pitch lines in the snapshot. Words like "one partner", "no agency layers", "senior operator", "ship-focused", "one point of contact" are SS marketing — never put them in the snapshot.
- No market statistics, no revenue projections, no valuations, no professional advice.
- If something was not covered, do not fill it in; record it under named_gaps.
- channel_shortlist may only contain channels the founder named or that follow directly from what they said. If the founder named no channels, leave it empty.
- next_steps should be concrete and, where possible, one per gap (prefer quick validation actions the founder can do themselves).
- idea_framing must describe the founder's idea in plain terms: what it is, who it is for, how far along they are. Nothing else.
Return the structured snapshot."""


def assemble(messages: List[dict], carried_gaps: List[str]) -> tuple[Snapshot, GroundingReport]:
    transcript = llm.transcript_text(messages)
    kb = "\n".join(retrieval_rail.all_grounding())
    sources = f"KNOWLEDGE BASE:\n{kb}\n\nTRANSCRIPT:\n{transcript}"

    if not llm.settings.has_llm:
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