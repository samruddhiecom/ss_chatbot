"""SS AI Advisor — conversation logic.

The bot is NOT an advisor. It does not give advice, tips, how-to, or diagnoses.
It has a natural, human conversation to understand the founder's situation and
what they care about, then guides them to a free strategy call with a real person.

Conversation phases (decided in graph.py, applied here):
  discover  -> be curious, learn their situation; returns reply + tappable chips; no call, no link
  cta       -> reflect their situation back, offer the call once, with the link
  objection -> answer their concern honestly, may re-offer the call once
  close     -> they signalled they're ready, give the booking link now
"""
from __future__ import annotations

import re

from app import llm
from app.kb import store
from app.schemas import DiscoverTurn, FounderProfile, ServiceRecommendation

OPENING = (
    "Hi \U0001f44b Tell me your business stage and biggest bottleneck, "
    "and I\u2019ll point you to the right service \u2014 or a human, if you want to take it further."
)

BOOK_URL = "https://simplified-startup-ui.vercel.app/#book"
PRICING_URL = "https://simplified-startup-ui.vercel.app/pricing"

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

_STRONG_BUYING_RE = re.compile(
    r"(send|share|drop|gimme|give me|grab me)\b.{0,20}\b(link|call|booking|slot|session)"
    r"|how do i\b.{0,20}\b(book|schedule|sign up|get started|get the link)"
    r"|book (the|a|my)\b.{0,10}\b(call|link|slot|session)"
    r"|schedule (the|a|my)\b.{0,10}\b(call|session|slot)"
    r"|sign me up|let'?s book|book a call|book the call",
    re.IGNORECASE,
)
_WEAK_AFFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|ok|okay|sounds good|let'?s do it|"
    r"let'?s go|i'?m in|go ahead|please do|do it|book it|perfect|great)\b",
    re.IGNORECASE,
)


def is_buying_signal(text: str, cta_offered: int = 0) -> bool:
    t = text or ""
    if _STRONG_BUYING_RE.search(t):
        return True
    if cta_offered > 0 and _WEAK_AFFIRM_RE.search(t):
        return True
    return False


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
- ready_for_cta: true ONLY when you clearly know ALL THREE of: (1) the founder's stage, (2) their specific
  bottleneck in their own words -- not a chip tap or a one-line answer, but something that tells you what is
  actually broken or missing in their business, AND (3) at least one of: what they have already tried, how
  urgent this is, or what their current setup looks like. A single exchange is never enough. If any of these
  three are missing or vague, set this false.
- follow_up: null (not used)

Rules:
- Never ask for revenue figures, budget, or funding status.
- bottleneck must be null until the founder has actually named a real problem -- a bare greeting or a vague
  opener is not a bottleneck.
- Base everything only on what the founder actually said."""


def extract_profile(messages: list[dict]) -> FounderProfile:
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


_BASE_PERSONA = (
    "You are the Simplified Startup AI Advisor, embedded on the Simplified Startup website.\n\n"
    "WHO YOU ARE: a warm, plain-spoken senior operator having a relaxed, coffee-chat style conversation with a\n"
    "founder. You are genuinely curious and human. You listen first and you sound like a real person, not a script.\n\n"
    "WHAT YOU DO NOT DO:\n"
    "- You do not give advice, tips, how-to guidance, steps, frameworks, or diagnoses. You never tell the founder\n"
    "  how to fix their problem. If you catch yourself explaining what they should do, stop and ask a question instead.\n"
    "- You never promise or imply results or outcomes. Never say things like 'we fix that', 'we turn traffic into\n"
    "  revenue', 'we'll get you results', 'specializes in solving', 'built to resolve'. Describe what Simplified\n"
    "  Startup does plainly; never claim what it will achieve for them.\n"
    "- You never quote prices or figures. For any pricing or cost question, point them to the pricing page: "
    + PRICING_URL + "\n"
    "- You never abbreviate the company name. Always write 'Simplified Startup' in full. Never write 'SS'.\n"
    "- You never use these words: leverage, synergistic, best-in-class, move the needle, holistically, unlock.\n\n"
    "STYLE:\n"
    "- Maximum 3 sentences. Warm and natural, never salesy or repetitive.\n"
    "- At most one question per message.\n"
    "- Never restart the conversation, never re-introduce yourself, and never repeat a point you already made.\n"
    "  Build on what the founder has told you and refer back to it in their own words.\n"
    "- Timeline questions: SEO usually takes 3 to 6 months for meaningful movement; paid ads and a new website\n"
    "  can move things in weeks. Say this plainly.\n"
    "- If asked whether you are human: say you are Simplified Startup's AI advisor and a real person can take it further."
)

_PHASE_DISCOVER = (
    "CURRENT GOAL -- GET TO KNOW THEM (do not sell yet):\n"
    "Do NOT offer, mention, or hint at the strategy call, and do NOT include any link.\n"
    "React briefly and warmly to what they just said, then ask ONE natural, specific question that moves you toward\n"
    "understanding their stage, their biggest bottleneck, or what they have already tried.\n\n"
    "Return two things:\n"
    "- reply: your short reply, ending in that one question.\n"
    "- chips: full conversational sentences written in the founder's own voice, as if they are answering the\n"
    "  question you just asked. Each chip must be 8 to 12 words, concrete and specific, never binary yes/no,\n"
    "  never one or two words. The number of chips should fit the question:\n"
    "    - 'what have you tried' type: 4 to 5 chips covering the most likely real answers\n"
    "    - 'what stage are you at' or 'what is your goal' type: 3 chips\n"
    "    - 'what specifically is broken' or a narrow follow-up: 2 to 3 chips\n"
    "  Write them as the founder would say it, not as labels. Good example for 'what have you tried':\n"
    "    ['We ran ads but burned through the budget fast',\n"
    "     'Mostly social posts but they do not convert to sales',\n"
    "     'We tried email campaigns but got very low click-through',\n"
    "     'We have not tried anything structured yet']\n"
    "  Bad example: ['Google Ads', 'Social media', 'Yes', 'No']\n"
    "  Never include an answer the founder has already given in this conversation.\n"
    "  Never mention any third-party tools, platforms, services, or competitor names in chips.\n"
    "  Never name Wix, WordPress, Squarespace, Shopify, Webflow, Framer, Fiverr, Upwork, or any other.\n"
    "  Describe the situation or outcome instead -- 'I tried a website builder but it did not match my brand'\n"
    "  not 'I tried Wix'. 'I hired someone but the project stalled' not 'I hired a freelancer on Fiverr'.\n"
    "  Do NOT add a 'Something else' chip -- that is handled by the frontend automatically."
)

_PHASE_CTA = (
    "CURRENT GOAL -- OFFER THE CALL, ONCE:\n"
    "You now understand their situation. In one or two sentences, reflect back the specific thing they care about\n"
    "using their own words, then warmly offer a free 30-minute strategy call with a real person from Simplified\n"
    "Startup. Include this exact link at the end: " + BOOK_URL + "\n"
    "Do NOT ask for a meeting time. Do NOT ask for their email. Just make the offer feel natural and share the link."
)

_PHASE_OBJECTION = (
    "CURRENT GOAL -- ANSWER THEM STRAIGHT:\n"
    "You have already offered the call once. Answer their question or concern honestly and briefly, using only what\n"
    "you actually know about Simplified Startup. Do not give advice or how-to. For pricing or cost questions, point\n"
    "them to " + PRICING_URL + "\n"
    "If it genuinely fits, you may offer the call one more time with this link: " + BOOK_URL + " -- but do not push,\n"
    "do not repeat yourself, and never ask for a meeting time or their email."
)

_PHASE_CLOSE = (
    "CURRENT GOAL -- THEY'RE READY, HAND THEM THE LINK:\n"
    "The founder has signalled they want to take the next step. Reply in exactly this shape and nothing else:\n"
    "one short warm sentence acknowledging it, then the sentence 'Book your free 30-minute strategy call here:'\n"
    "followed by this exact link: " + BOOK_URL + "\n"
    "Do NOT greet them, do NOT re-introduce yourself, do NOT ask a question, do NOT ask for their email or a meeting\n"
    "time, and do NOT invent any other process (no 'we'll send you a draft', no forms)."
)

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


def _user_content(profile: FounderProfile, kb_text: str, transcript: str) -> str:
    return (
        f"WHAT YOU KNOW ABOUT THIS FOUNDER SO FAR: {_profile_summary(profile)}\n\n"
        f"REFERENCE INFORMATION about Simplified Startup (use only if relevant, never quote prices):\n"
        f"{kb_text}\n\n"
        f"CONVERSATION SO FAR:\n{transcript}"
    )


def generate_reply(messages: list[dict], kb_chunks: list[str], phase: str, profile: FounderProfile):
    """Returns (reply_text, chips_list). chips are only produced in the discover phase."""
    if not llm.settings.has_llm:
        return ("Tell me a bit about what you're building and where you're stuck.", [])

    transcript = llm.transcript_text(messages)
    kb_text = "\n\n".join(kb_chunks) if kb_chunks else ""
    user = _user_content(profile, kb_text, transcript)

    if phase == "discover":
        system = _BASE_PERSONA + "\n\n" + _PHASE_DISCOVER
        try:
            out = llm.structured(
                system=system,
                user=user,
                response_model=DiscoverTurn,
                temperature=0.4,
            )
            chips = [c.strip() for c in (out.chips or []) if c and c.strip()][:6]
            reply = (out.reply or "").strip()
            if reply:
                return (reply, chips)
        except Exception:
            pass
        text = llm.generate(system, user, temperature=0.4)
        return (text, [])

    system = _BASE_PERSONA + "\n\n" + _PHASES.get(phase, _PHASE_DISCOVER)
    text = llm.generate(system, user, temperature=0.4)
    return (text, [])


_REC_SYSTEM = """You produce a Simplified Startup service recommendation based on what the founder told you.

Rules:
- primary_service must be one of the 8 Simplified Startup services exactly as named.
- reasoning must be grounded in what the founder said -- no invented facts.
- one_piece_of_advice must be concrete and actionable.
- cta is the free strategy call offer -- warm, no pressure.
- page_link is the most relevant Simplified Startup page from the KB.
- Never quote prices. Never promise results.
- Always say "Simplified Startup" in full -- never abbreviate to "SS".
- secondary_service only if clearly relevant -- leave null otherwise."""

_SERVICES_LIST = ", ".join(SERVICE_LINKS.keys())


def build_recommendation(messages: list[dict], profile: FounderProfile) -> ServiceRecommendation:
    if not llm.settings.has_llm:
        return ServiceRecommendation(
            primary_service="Business and Startup Advisory",
            reasoning="Based on your stage and bottleneck.",
            one_piece_of_advice="Start with a clear go-to-market plan before investing in any channel.",
            cta="Book a free 30-minute strategy call -- you'll leave with a written plan either way.",
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