"""All data contracts in one place so every module agrees on shapes.

The Snapshot model is the durable deliverable. The ConversationState is threaded
through the LangGraph state machine, one invocation per user turn.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
class Stage(str, Enum):
    IDEA = "idea"
    MARKET_CUSTOMER = "market_customer"
    OFFER_CHANNELS = "offer_channels"
    OPERATIONS = "operations"
    DONE = "done"


STAGE_ORDER: List[Stage] = [
    Stage.IDEA,
    Stage.MARKET_CUSTOMER,
    Stage.OFFER_CHANNELS,
    Stage.OPERATIONS,
]


def next_stage(stage: Stage) -> Stage:
    if stage == Stage.DONE:
        return Stage.DONE
    idx = STAGE_ORDER.index(stage)
    if idx + 1 < len(STAGE_ORDER):
        return STAGE_ORDER[idx + 1]
    return Stage.DONE


# --------------------------------------------------------------------------- #
# Intent (input rail)
# --------------------------------------------------------------------------- #
class Intent(str, Enum):
    ON_TOPIC = "on_topic"
    ADVICE_LEGAL = "advice_legal"
    ADVICE_TAX = "advice_tax"
    ADVICE_FINANCIAL = "advice_financial"
    ADVICE_INVESTMENT = "advice_investment"
    PROJECTION_BAIT = "projection_bait"
    STATISTICS_BAIT = "statistics_bait"
    INJECTION = "injection"
    OFF_TOPIC = "off_topic"
    ABUSE = "abuse"
    HARDSHIP = "hardship"


# Intents that trigger the deferral pattern and drop a named gap into the snapshot.
DEFERRAL_INTENTS = {
    Intent.ADVICE_LEGAL,
    Intent.ADVICE_TAX,
    Intent.ADVICE_FINANCIAL,
    Intent.ADVICE_INVESTMENT,
    Intent.PROJECTION_BAIT,
    Intent.STATISTICS_BAIT,
}

# Intents that produce a bounded response but no snapshot content.
NON_PROGRESSING_INTENTS = DEFERRAL_INTENTS | {
    Intent.INJECTION,
    Intent.OFF_TOPIC,
    Intent.ABUSE,
    Intent.HARDSHIP,
}


class IntentDecision(BaseModel):
    """Structured output of the classifier (Instructor-parsed)."""
    intent: Intent = Field(description="The single best-matching intent for the user's latest message.")
    reason: str = Field(description="One short clause explaining the choice.")


# --------------------------------------------------------------------------- #
# Stage extraction (slot filling)
# --------------------------------------------------------------------------- #
class StageExtraction(BaseModel):
    """What the model organised from the founder's input for the current stage.

    Only fields the founder actually provided should be filled. Never invent.
    """
    captured: dict = Field(
        default_factory=dict,
        description="Key facts the founder stated for this stage, organised. Keys are short labels.",
    )
    covered: bool = Field(description="True if the stage has enough to move on.")
    follow_up: Optional[str] = Field(
        default=None,
        description="If not covered, one short follow-up question targeting the biggest missing piece.",
    )


# --------------------------------------------------------------------------- #
# Snapshot (the durable deliverable)
# --------------------------------------------------------------------------- #
class Snapshot(BaseModel):
    idea_framing: str = Field(description="A crisp framing of the idea and stage, organised from what the founder said.")
    customer_hypothesis: str = Field(description="Who the customer is, as the founder understands them. No invented segments.")
    offer_sketch: str = Field(description="The offer as described by the founder.")
    channel_shortlist: List[str] = Field(default_factory=list, description="Channels the founder named or that follow directly from what they said.")
    named_gaps: List[str] = Field(default_factory=list, description="What is unknown or unvalidated, including anything deferred to a professional.")
    next_steps: List[str] = Field(default_factory=list, description="Concrete next actions, one per gap where possible.")


class GroundingReport(BaseModel):
    """Output of the claims-clean self-check on synthesised text."""
    supported: bool = Field(description="True if every claim traces to the provided sources.")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims not supported by sources.")


# --------------------------------------------------------------------------- #
# Conversation state (threaded through the graph)
# --------------------------------------------------------------------------- #
class Turn(TypedDict):
    role: str  # "user" | "assistant"
    content: str


class GraphState(TypedDict, total=False):
    session_id: str
    messages: List[Turn]
    last_user_input: str
    current_stage: str          # Stage value
    intent: str                 # Intent value of last input
    stage_followups: dict       # stage -> count
    captured: dict              # stage -> captured dict
    assistant_reply: str
    named_gaps: List[str]
    grounding_flags: List[str]
    snapshot_ready: bool


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #
class StartResponse(BaseModel):
    session_id: str
    disclosure: List[str]
    message: str
    stage: str


class SessionRequest(BaseModel):
    session_id: str


class MessageRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    message: str
    stage: str
    intent: str
    snapshot_ready: bool
    grounding_flags: List[str] = Field(default_factory=list)


class SnapshotResponse(BaseModel):
    snapshot: Snapshot
    grounding_report: GroundingReport


class CaptureRequest(BaseModel):
    session_id: str
    name: str
    email: str


class CaptureResponse(BaseModel):
    captured: bool
    tool_source: str
    snapshot_attached: bool
