"""All data contracts for the SS AI Advisor chatbot.

The bot has a natural, human conversation to understand a founder's situation and
what they care about, then guides them to a free strategy call with a real person.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --------------------------------------------------------------------------- #
# Conversation state
# --------------------------------------------------------------------------- #
class ConvStage(str, Enum):
    OPENING = "opening"
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    CTA = "cta"
    DONE = "done"


# --------------------------------------------------------------------------- #
# Intent (input rail)
# --------------------------------------------------------------------------- #
class Intent(str, Enum):
    ON_TOPIC = "on_topic"
    GREETING = "greeting"
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
    EXISTING_CLIENT = "existing_client"


DEFERRAL_INTENTS = {
    Intent.ADVICE_LEGAL,
    Intent.ADVICE_TAX,
    Intent.ADVICE_FINANCIAL,
    Intent.ADVICE_INVESTMENT,
    Intent.PROJECTION_BAIT,
    Intent.STATISTICS_BAIT,
}

NON_PROGRESSING_INTENTS = DEFERRAL_INTENTS | {
    Intent.GREETING,
    Intent.INJECTION,
    Intent.OFF_TOPIC,
    Intent.ABUSE,
    Intent.HARDSHIP,
    Intent.EXISTING_CLIENT,
}


class IntentDecision(BaseModel):
    intent: Intent = Field(description="The single best-matching intent.")
    reason: str = Field(description="One short clause explaining the choice.")


# --------------------------------------------------------------------------- #
# Founder profile (built up during conversation)
# --------------------------------------------------------------------------- #
class FounderProfile(BaseModel):
    stage: Optional[str] = Field(default=None, description="idea / pre-revenue / early revenue / scaling")
    bottleneck: Optional[str] = Field(default=None, description="Biggest bottleneck in their own words")
    already_tried: Optional[str] = Field(default=None, description="What they have already tried")
    service_interest: Optional[str] = Field(default=None, description="Which SS service they seem to need")
    covered: bool = Field(default=False, description="True if we have enough to make a recommendation")
    ready_for_cta: bool = Field(
        default=False,
        description="True only if the founder has shared enough (stage + real bottleneck) that offering the "
        "free strategy call will land as helpful rather than premature.",
    )
    follow_up: Optional[str] = Field(default=None, description="Not used.")


class DiscoverTurn(BaseModel):
    reply: str = Field(description="The assistant's warm, 1–3 sentence reply.")
    chips: List[str] = Field(
        default_factory=list,
        description="2–4 short tappable answer options that each answer the question just asked.",
    )


# --------------------------------------------------------------------------- #
# Service recommendation
# --------------------------------------------------------------------------- #
class ServiceRecommendation(BaseModel):
    primary_service: str = Field(description="One of the 8 Simplified Startup services, exact name.")
    secondary_service: Optional[str] = Field(default=None, description="Only if clearly relevant.")
    reasoning: str = Field(description="Grounded in what the founder said. No invented facts.")
    one_piece_of_advice: str = Field(description="Concrete and actionable.")
    cta: str = Field(description="Warm free-strategy-call offer, no pressure.")
    page_link: str = Field(description="Relative path to the most relevant SS page from KB.")


# --------------------------------------------------------------------------- #
# LangGraph state
# --------------------------------------------------------------------------- #
class GraphState(TypedDict, total=False):
    session_id: str
    messages: List[dict]
    last_user_input: str
    intent: str
    conv_stage: str
    founder_profile: dict
    retrieved_chunks: List[str]
    assistant_reply: str
    chips: List[str]
    recommendation: dict
    recommendation_ready: bool
    cta_ready: bool
    cta_offered: int
    grounding_flags: List[str]
    # New: lead + session timestamps carried through for conversation persistence
    started_at: str
    lead_name: str
    lead_email: str


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #
class StartResponse(BaseModel):
    session_id: str
    disclosure: List[str]
    message: str


class MessageRequest(BaseModel):
    session_id: str
    message: str


class SessionRequest(BaseModel):
    session_id: str


class MessageResponse(BaseModel):
    message: str
    intent: str
    recommendation_ready: bool
    chips: List[str] = Field(default_factory=list)
    grounding_flags: List[str] = Field(default_factory=list)
    # New: lets the frontend decide which buttons / CTA shape to render.
    phase: str = Field(default="")
    show_explore_button: bool = Field(default=False)
    show_success_stories_button: bool = Field(default=False)


class RecommendationResponse(BaseModel):
    recommendation: ServiceRecommendation


class CaptureRequest(BaseModel):
    session_id: str
    name: str
    email: str


class CaptureResponse(BaseModel):
    captured: bool
    tool_source: str
    recommendation_attached: bool


# --------------------------------------------------------------------------- #
# Grounding report (used by output rail)
# --------------------------------------------------------------------------- #
class GroundingReport(BaseModel):
    supported: bool = Field(description="True if every claim traces to the provided sources.")
    unsupported_claims: List[str] = Field(default_factory=list)
    corrections: List[str] = Field(default_factory=list)
