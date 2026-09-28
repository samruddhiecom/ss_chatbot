"""SS AI Advisor — conversation logic.

The bot is NOT an advisor. It does not give advice, tips, how-to, or diagnoses.
It has a natural, human conversation to understand the founder's situation and
what they care about, then guides them to a free strategy call with a real person.

Conversation phases (decided in graph.py, applied here):
  discover  -> be curious, learn their situation, no call, no link
  cta       -> reflect their situation back, offer the call once, with the link
  objection -> answer their concern honestly, may re-offer the call once
  close     -> they signalled they're ready, give the booking link now
"""
from __future__ import annotations

import re

from app import llm
from app.kb import store
from app.schemas import FounderProfile, ServiceRecommendation

# ── Opening message (shown by the frontend on load) ──────────────────────────
OPENING = (
    "Hi \U0001f44b Tell me your business stage and biggest bottleneck, "
    "and I\u2019ll point you to the right service \u2014 or a human, if you want to take it further."
)

# ── Links ────────────────────────────────────────────────────────────────────
BOOK_URL = "https://simplified-startup-ui.vercel.app/#book"
PRICING_URL = "https://simplified-startup-ui.vercel.app/pricing"

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

# ── Buying-signal detection (deterministic) ──────────────────────────────────
# A short affirmation, or an explicit request to book / get the link.
_BUYING_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|ok|okay|sounds good|"
    r"let'?s do it|let'?s go|i'?m in|go ahead|please do|do it|"
    r"sign me up|book it|perfect|great)\b"
    r"|(send|share|drop|gimme|give me|grab me)\b.{0,20}\b(link|call|booking|slot|session)"
    r"|how do i\b.{0,20}\b(get|book|start|sign|schedule|join)"
    r"|book (the|a|my)\b.{0,10}\b(call|link|slot|session)"
    r"|schedule (the|a|my)\b.{0,10}\b(call|session|slot)",
    re.IGNORECASE,
)


def is_buying_signal(text: str) -> bool:
    return bool(_BUYING_RE.search(text or ""))


# ── Profile extraction ────────────────────────────────────────────────────────
_PROFILE_SYSTEM = """You extract a founder's profile from a conversation for the Simplified Startup AI Advisor.

Extract:
- stage: one of idea / pre-revenue / early revenue / scaling (or null if not clear)
- bottleneck: their single biggest problem in their own words (or null if they have not said one yet)
- already_tried: what they have already attempted (or null)
- service_interest: which of these Simplified Startup services they seem to need most:
  Digital Marketing, Website Development, Branding and Growth,
  Sales and Lead Generation, AI Automation, Business and Startup Advisory,
  Talent and Staffing, Bookkeeping and Accounting (or null if unclear)
- covered: leave false unless they have clearly stated a real bottleneck
- follow_up: null (not used)

Rules:
- Never ask for revenue figures, budget, or funding status.
- bottleneck must be null until the founder has actually named a real problem — a bare greeting or a vague
  opener is not a bottleneck.
- Base everything only on what the founder actually said."""


def extract_profile(messages: list[dict]) -> FounderProfile:
    """Re-read the founder from the whole conversation each turn, so the profile
    reflects everything said so far (not just the first message)."""
    if not llm.settings.has_llm:
        return FounderProfile()
    transcript = llm.transcript_text(messages)
    return llm.structured(
        system=_PROFILE_SYSTEM,
        user=transcript,
        response_model=FounderProfile,
        model=llm.settings.fast_model,
        temperature=0.1,
    )


# ── Reply generation ──────────────────────────────────────────────────────────
_BASE_PERSONA = f"""You are the Simplified Startup AI Advisor, embedded on the Simplified Startup website.

WHO YOU ARE: a warm, plain-spoken senior operator having a relaxed, coffee-chat style conversation with a
founder. You are genuinely curious and human. You listen first and you sound like a real person, not a script.

WHAT YOU DO NOT DO:
- You do not give advice, tips, how-to guidance, steps, frameworks, or diagnoses. You never tell the founder
  how to fix their problem. If you catch yourself explaining what they should do, stop and ask a question instead.
- You never promise or imply results or outcomes. Never say things like "we fix that", "we turn traffic into
  revenue", "we'll get you results", "specializes in solving", "built to resolve". Describe what Simplified
  Startup does plainly; never claim what it will achieve for them.
- You never quote prices or figures. For any pricing or cost question, point them to the pricing page: {PRICING_URL}
- You never abbreviate the company name. Always write "Simplified Startup" in full. Never write "SS".
- You never use these words: leverage, synergistic, best-in-class, move the needle, holistically, unlock.

STYLE:
- Maximum 3 sentences. Warm and natural, never salesy or repetitive.
- At most one question per message.
- Never restart the conversation or repeat a point you already made. Build on what the founder has told you and
  refer back to it in their own words.
- Timeline questions: SEO usually takes 3 to 6 months for meaningful movement; paid ads and a new website can
  move things in weeks. Say this plainly.
- If asked whether you are human: say you are Simplified Startup's AI advisor and a real person can take it further.
"""

_PHASE_DISCOVER = """CURRENT GOAL — GET TO KNOW THEM:
Do NOT offer, mention, or hint at the strategy call yet, and do NOT include any link.
React like a real person to what they just said, then ask ONE natural, specific question that helps you
understand their stage, their biggest bottleneck, or what they are most eager to sort out. You are learning
about them, not pitching. Keep it light and genuinely curious."""

_PHASE_CTA = f"""CURRENT GOAL — OFFER THE CALL, ONCE:
You now understand their situation. In one or two sentences, reflect back the specific thing they care about
using their own words, then warmly offer a free 30-minute strategy call with a real person from Simplified
Startup. Include this exact link at the end: {BOOK_URL}
Do NOT ask for a meeting time. Do NOT ask for their email. Just make the offer feel natural and share the link."""

_PHASE_OBJECTION = f"""CURRENT GOAL — ANSWER THEM STRAIGHT:
You have already offered the call once. Answer their question or concern honestly and briefly, using only what
you actually know about Simplified Startup. Do not give advice or how-to. For pricing or cost questions, point
them to {PRICING_URL}
If it genuinely fits, you may offer the call one more time with this link: {BOOK_URL} — but do not push, do not
repeat yourself, and never ask for a meeting time or their email."""

_PHASE_CLOSE = f"""CURRENT GOAL — THEY'RE READY, HAND THEM THE LINK:
The founder has signalled they want to take the next step. Warmly give them the booking link now: {BOOK_URL}
Do NOT ask for their email, do NOT ask for a meeting time, and do NOT invent any other process (no "we'll send
you a draft", no forms). One or two sentences, then the link."""

_PHASES = {
    "discover": _PHASE_DISCOVER,
    "cta": _PHASE_CTA,
    "objection": _PHASE_OBJECTION,
    "close": _PHASE_CLOSE,
}


def _profile_summary(profile: FounderProfile) -> str:
    bits = []
    if profile.stage:
        bits.append(f"stage: {profile.stage}")
    if profile.bottleneck:
        bits.append(f"bottleneck: {profile.bottleneck}")
    if profile.already_tried:
        bits.append(f"already tried: {profile.already_tried}")
    if profile.service_interest:
        bits.append(f"likely relevant service: {profile.service_interest}")
    return "; ".join(bits) if bits else "nothing concrete yet"


def generate_reply(messages: list[dict], kb_chunks: list[str], phase: str, profile: FounderProfile) -> str:
    if not llm.settings.has_llm:
        return "Tell me a bit about what you're building and where you're stuck."
    transcript = llm.transcript_text(messages)
    kb_text = "\n\n".join(kb_chunks) if kb_chunks else ""
    system = _BASE_PERSONA + "\n\n" + _PHASES.get(phase, _PHASE_DISCOVER)
    user = (
        f"WHAT YOU KNOW ABOUT THIS FOUNDER SO FAR: {_profile_summary(profile)}\n\n"
        f"REFERENCE INFORMATION about Simplified Startup (use only if relevant, never quote prices):\n"
        f"{kb_text}\n\n"
        f"CONVERSATION SO FAR:\n{transcript}"
    )
    return llm.generate(system, user, temperature=0.4)


# ── Recommendation assembly (used by the /recommendation path) ───────────────
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