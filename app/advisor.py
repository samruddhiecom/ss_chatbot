"""SS AI Advisor — conversation logic.

Strictly follows the KB document (Sections 13-16):
- Job: understand stage + bottleneck, give one useful directional insight,
  point to the right SS service, offer the free strategy call once.
- Persona: senior operator on a coffee chat. Warm, plain-spoken, brief.
- Style: 2-5 sentences per reply. Never more than one question per message.
- Mention the strategy call at most twice per conversation.
"""
from __future__ import annotations

from app import llm
from app.kb import store
from app.schemas import FounderProfile, ServiceRecommendation

# ── Opening message (from KB Section 13) ────────────────────────────────────
OPENING = (
    "Hi \U0001f44b Tell me your business stage and biggest bottleneck, "
    "and I\u2019ll point you to the right service \u2014 or a human, if you want to take it further."
)

# ── Service page links (from KB Section 10) ──────────────────────────────────
SERVICE_LINKS = {
    "Digital Marketing": "/digital-marketing",
    "Website Development": "/website-development",
    "Branding and Growth": "/branding-growth",
    "Sales and Lead Generation": "/sales-lead-gen",
    "AI Automation": "/ai-automation",
    "Business and Startup Advisory": "/business-advisory",
    "Talent and Staffing": "/talent-staffing",
    "Bookkeeping and Accounting": "/bookkeeping",
}

# ── Profile extraction ────────────────────────────────────────────────────────
_PROFILE_SYSTEM = """You extract a founder's profile from a conversation for the Simplified Startup AI Advisor.

Extract:
- stage: one of idea / pre-revenue / early revenue / scaling (or null if not clear)
- bottleneck: their biggest problem in their own words (or null)
- already_tried: what they have already attempted (or null)
- service_interest: which of these Simplified Startup services they seem to need most:
  Digital Marketing, Website Development, Branding and Growth,
  Sales and Lead Generation, AI Automation, Business and Startup Advisory,
  Talent and Staffing, Bookkeeping and Accounting (or null if unclear)
- covered: true if you have ANY useful context about their business situation — even just a bottleneck or a problem they mentioned. Set this to true after the first substantive message from the founder.
- follow_up: only ask a follow-up if the founder has given you literally nothing to work with

Rules:
- Never ask for revenue figures, budget, or funding status
- covered should be true after the first real message — err on the side of true
- follow_up must be one question only, plain and direct"""


def extract_profile(messages: list[dict]) -> FounderProfile:
    if not llm.settings.has_llm:
        return FounderProfile(covered=True)
    transcript = llm.transcript_text(messages)
    return llm.structured(
        system=_PROFILE_SYSTEM,
        user=transcript,
        response_model=FounderProfile,
        temperature=0.1,
    )


# ── Reply generation ──────────────────────────────────────────────────────────
_REPLY_SYSTEM = """You are the Simplified Startup AI Advisor. Your ONLY job is to get the founder to book a free strategy call with Simplified Startup.

You are NOT an advisor. You do NOT give advice. You do NOT explain how to fix things.

Your approach:
1. Acknowledge what the founder said — show you understand their situation.
2. Signal that this is exactly what Simplified Startup handles — briefly, without pitching.
3. Create mild urgency or curiosity — pull them toward the call.
4. Ask ONE question that moves them closer to booking, OR offer the call directly.

When you have enough context, close with the call:
"Book a free 30-minute strategy call — you'll leave with a written plan either way: simplified-startup-ui.vercel.app/#book"

Rules:
- Maximum 3 sentences per reply.
- Never give advice, tips, steps, or how-to guidance.
- Never list services or explain what Simplified Startup does in detail.
- Always say "Simplified Startup" in full — never abbreviate to "SS".
- Never use jargon: leverage, synergistic, best-in-class, move the needle, holistically, unlock.
- Never quote prices.
- Never promise results.
- One question per message maximum.
- Warm, direct, confident. Like a senior operator who has seen this problem before and knows exactly what to do.
- Do not pretend to be human if asked.
- Existing clients go to simplifiedstartupllc@gmail.com immediately.

The goal of every single reply is to get them to book the call."""


def generate_reply(messages: list[dict], kb_chunks: list[str], cta_count: int) -> str:
    if not llm.settings.has_llm:
        return "Tell me your stage and biggest bottleneck and I'll point you in the right direction."
    transcript = llm.transcript_text(messages)
    kb_text = "\n\n".join(kb_chunks) if kb_chunks else ""
    cta_note = f"\n\nNote: You have already offered the strategy call {cta_count} time(s). {'Do NOT mention it again.' if cta_count >= 2 else 'You may mention it if it naturally fits.'}"
    return llm.generate(
        system=_REPLY_SYSTEM + cta_note,
        user=f"KNOWLEDGE BASE:\n{kb_text}\n\nCONVERSATION:\n{transcript}",
        temperature=0.3,
    )


# ── Recommendation assembly ───────────────────────────────────────────────────
_REC_SYSTEM = """You produce a Simplified Startup service recommendation based on what the founder told you.

Rules:
- primary_service must be one of the 8 Simplified Startup services exactly as named.
- reasoning must be grounded in what the founder said — no invented facts.
- one_piece_of_advice must be concrete and actionable.
- cta is the free strategy call offer — warm, no pressure.
- page_link is the most relevant Simplified Startup page from the KB.
- Never quote prices. Never promise results.
- Always say "Simplified Startup" in full — never abbreviate to "SS".
- secondary_service only if clearly relevant — leave null otherwise."""

_SERVICES_LIST = ", ".join(SERVICE_LINKS.keys())


def build_recommendation(messages: list[dict], profile: FounderProfile) -> ServiceRecommendation:
    if not llm.settings.has_llm:
        return ServiceRecommendation(
            primary_service="Business and Startup Advisory",
            reasoning="Based on your stage and bottleneck.",
            one_piece_of_advice="Start with a clear go-to-market plan before investing in any channel.",
            cta="Book a free 30-minute strategy call — you'll leave with a written plan either way.",
            page_link="/business-advisory",
        )
    transcript = llm.transcript_text(messages)
    kb_chunks = store.query(profile.bottleneck or "startup advisory", top_k=6)
    kb_text = "\n\n".join(kb_chunks)
    return llm.structured(
        system=f"{_REC_SYSTEM}\n\nAvailable Simplified Startup services: {_SERVICES_LIST}\n\nSimplified Startup page links: {SERVICE_LINKS}",
        user=f"KNOWLEDGE BASE:\n{kb_text}\n\nCONVERSATION:\n{transcript}\n\nFounder stage: {profile.stage}\nBottleneck: {profile.bottleneck}",
        response_model=ServiceRecommendation,
        temperature=0.2,
    )