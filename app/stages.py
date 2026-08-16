"""The four planning stages: openers, per-stage extraction, and follow-up generation.

Each on-topic turn extracts what the founder provided for the current stage, decides
whether the stage is covered, and either advances or asks one targeted follow-up.
"""
from __future__ import annotations

from typing import List

from app import llm, voice
from app.rails import retrieval_rail
from app.schemas import Stage, StageExtraction

STAGE_LABEL = {
    Stage.IDEA.value: "idea and stage",
    Stage.MARKET_CUSTOMER.value: "market and customer",
    Stage.OFFER_CHANNELS.value: "offer and channels",
    Stage.OPERATIONS.value: "operations",
}

OPENERS = {
    Stage.IDEA.value: "Let's start simple. What are you building, and how far along are you?",
    Stage.MARKET_CUSTOMER.value: "Who's it for, and what do you know about them so far?",
    Stage.OFFER_CHANNELS.value: "What exactly are you offering, and how do you plan to reach people?",
    Stage.OPERATIONS.value: "Last piece: how does it actually get delivered? Team, tools, key steps.",
}

_EXTRACT_SYSTEM = """You organise a founder's input for one planning stage.
Capture ONLY what they actually said, as short labelled facts. Invent nothing.
Decide if the stage has enough to move on. If not, propose ONE short follow-up
question targeting the single most important missing piece. Do not ask for
statistics, projections, or professional-advice topics."""


def opener(stage: str) -> str:
    return OPENERS.get(stage, "Tell me more.")


def extract(stage: str, stage_messages: List[dict]) -> StageExtraction:
    if not llm.settings.has_llm:
        return StageExtraction(captured={}, covered=True, follow_up=None)
    convo = llm.transcript_text(stage_messages)
    guidance = "\n".join(retrieval_rail.for_stage(stage, STAGE_LABEL.get(stage, stage)))
    return llm.structured(
        system=f"{_EXTRACT_SYSTEM}\n\nStage guidance:\n{guidance}",
        user=f"Stage: {STAGE_LABEL.get(stage, stage)}\n\nConversation so far in this stage:\n{convo}",
        response_model=StageExtraction,
        temperature=0.1,
    )


def acknowledge_and_advance(stage: str, next_label: str) -> str:
    """A short grounded bridge into the next stage."""
    if not llm.settings.has_llm:
        return f"Got it. Next, let's cover {next_label}."
    raw = llm.generate(
        system=(
            "You are SS's planning assistant. Briefly acknowledge what the founder just covered "
            "(no new facts) and transition to the next stage in one short line."
        ),
        user=f"Just finished: {STAGE_LABEL.get(stage, stage)}. Next stage: {next_label}.",
        temperature=0.3,
    )
    return voice.apply_voice(raw)


def styled_followup(follow_up: str) -> str:
    return voice.apply_voice(follow_up) if follow_up else "Tell me a bit more."
