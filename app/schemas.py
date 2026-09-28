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
    OPENING = "opening"        # gathering stage + bottleneck
    CLARIFY = "clarify"        # one follow-up if needed
    RECOMMEND = "recommend"    # service recommendation given
    CTA = "cta"                # call offered
    DONE = "done"              # lead captured


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

# Intents that do not go to the advisor node — they get a fixed, cheap reply.
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
    follow_up: Optional[str] = Field(default=None, description="One follow-up question if not yet covered")


# --------------------------------------------------------------------------- #
# Service recommendation (used by the /recommendation path)
# --------------------------------------------------------------------------- #
class ServiceRecommendation(BaseModel):
    primary_service: str = Field(description="The single most relevant SS service for this founder.")
    secondary_service: Optional[str] = Field(default=None, description="A second SS service if clearly relevant.")
    reasoning: str = Field(description="Why this service fits — grounded in what the founder said.")
    one_piece_of_advice: str = Field(description="One concrete directional insight for this founder.")
    cta: str = Field(description="The free strategy call offer — once, no pressure.")
    page_link: Optional[str] = Field(default=None, description="The most relevant SS page link from the KB.")


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class Turn(TypedDict):
    role: str
    content: str


class GraphState(TypedDict, total=False):
    session_id: str
    messages: List[Turn]
    last_user_input: str
    conv_stage: str
    intent: str
    founder_profile: dict
    assistant_reply: str
    recommendation: dict
    recommendation_ready: bool
    cta_ready: bool
    cta_offered: int
    grounding_flags: List[str]


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
    grounding_flags: List[str] = Field(default_factory=list)


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