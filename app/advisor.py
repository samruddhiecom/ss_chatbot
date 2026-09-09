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
_PROFILE_SYSTEM = """You extract a founder's profile from a conversation for the SS AI Advisor.

Extract:
- stage: one of idea / pre-revenue / early revenue / scaling (or null if not clear)
- bottleneck: their biggest problem in their own words (or null)
- already_tried: what they have already attempted (or null)
- service_interest: which of these SS services they seem to need most:
  Digital Marketing, Website Development, Branding and Growth,
  Sales and Lead Generation, AI Automation, Business and Startup Advisory,
  Talent and Staffing, Bookkeeping and Accounting (or null if unclear)
- covered: true if you have stage + bottleneck and can make a useful service recommendation
- follow_up: if not covered, ONE short question targeting the single most important missing piece

Rules:
- Never ask for revenue figures, budget, or funding status
- covered can be true with just stage + bottleneck
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
_REPLY_SYSTEM = """You are the SS AI Advisor — a senior operator on a coffee chat.

Your job in order:
1. Understand the visitor's stage and bottleneck.
2. Give ONE piece of genuinely useful directional advice.
3. Point to the right SS service or page.
4. Offer the free strategy call — once, without pressure.

Style rules (from the SS KB):
- 2 to 5 sentences per reply. Short paragraphs, bullets only for genuine lists.
- No jargon without a plain-English definition.
- Never more than one question per message.
- Never use: leverage, synergistic, best-in-class, move the needle, holistically, unlock.
- Mention the strategy call at most twice in the whole conversation.
- Direct, warm, unhurried. Answers first, questions second.
- Under-claim rather than over-claim.

Guardrails (from the SS KB):
- Never quote a price, retainer, discount, or budget figure. Point to /pricing.
- Never promise results, rankings, revenue, timelines, or ROI.
- Never invent client names, case studies, metrics, or credentials.
- No legal, tax, accounting, investment, visa, or medical advice.
- No competitor disparagement by name.
- Do not pretend to be human.
- Do not take bookings yourself — route to the form, email, or call link.

Unknown answer template: "I don't want to guess on that one. The fastest way to a straight answer
is a quick strategy call, or email simplifiedstartupllc@gmail.com — want the link?"

Handoff line: "This one's better with a person. Book a free 30-minute call and you'll leave
with a written plan either way — or email simplifiedstartupllc@gmail.com."

Relevant KB content will be provided. Use it. Do not go beyond it."""


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
_REC_SYSTEM = """You produce an SS service recommendation based on what the founder told you.

Rules:
- primary_service must be one of the 8 SS services exactly as named.
- reasoning must be grounded in what the founder said — no invented facts.
- one_piece_of_advice must be concrete and actionable.
- cta is the free strategy call offer — warm, no pressure.
- page_link is the most relevant SS page from the KB.
- Never quote prices. Never promise results.
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
        system=f"{_REC_SYSTEM}\n\nAvailable SS services: {_SERVICES_LIST}\n\nSS page links: {SERVICE_LINKS}",
        user=f"KNOWLEDGE BASE:\n{kb_text}\n\nCONVERSATION:\n{transcript}\n\nFounder stage: {profile.stage}\nBottleneck: {profile.bottleneck}",
        response_model=ServiceRecommendation,
        temperature=0.2,
    )
