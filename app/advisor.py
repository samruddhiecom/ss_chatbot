"""SS AI Advisor -- conversation logic.

The bot is NOT an advisor. It does not give advice, tips, how-to, or diagnoses.
It has a natural, human conversation to understand the founder's situation and
what they care about, then guides them to a free strategy call with a real person.

Conversation phases (decided in graph.py, applied here):
  discover    -> be curious, learn their situation; returns reply + tappable chips; no call, no link
  cta         -> reflect their situation back, offer the call once, with the link
  soft_close  -> founder is unsure; offer the call AND a path to explore the main website
  objection   -> answer their concern honestly, may re-offer the call once
  close       -> they signalled they're ready, give the booking link now
  success     -> founder asked for proof / case studies; point to success stories page
"""
from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field

from app import llm
from app.kb import store
from app.schemas import DiscoverTurn, FounderProfile, ServiceRecommendation

OPENING = (
    "Hi \U0001f44b Tell me your business stage and biggest bottleneck, "
    "and I\u2019ll point you to the right service \u2014 or a human, if you want to take it further."
)

BOOK_URL = "https://simplified-startup-ui.vercel.app/#book"
PRICING_URL = "https://simplified-startup-ui.vercel.app/pricing"
WEBSITE_URL = "https://simplified-startup-ui.vercel.app"
SUCCESS_STORIES_URL = "https://simplified-startup-ui.vercel.app/success-stories"

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
    r"|(set up|setup|set-up)\b.{0,10}\b(a |the |my )?(call|meeting|session|time|chat)"
    r"|let'?s (set up|setup|do|schedule|book|talk|chat|hop on)\b.{0,15}\b(call|meeting|session|time|chat|it up)?"
    r"|sign me up|let'?s book|book a call|book the call"
    r"|just book|book us in|let's get started"
    r"|i'?m ready (to|for)\b.{0,20}\b(call|start|book|go|move forward|chat|talk)"
    r"|we'?re ready (to|for)\b.{0,20}\b(call|start|book|go|move forward|chat|talk)"
    r"|i want to (get started with you|work with you|move forward with|book|schedule|sign up)"
    r"|let'?s (do it|go)\b",
    re.IGNORECASE,
)
# Weak affirm only counts as a buying signal when the WHOLE message is the affirmation
# (so "ok sorry, how long before results?" does NOT count — that is a question, not a yes).
_WEAK_AFFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|ok|okay|sounds good|let'?s do it|"
    r"let'?s go|i'?m in|go ahead|please do|do it|book it|perfect|great)"
    r"[\s!.,]*$",
    re.IGNORECASE,
)

# Loss / hardship signals that call for a one-line of practical empathy BEFORE the normal reply.
_EMPATHY_RE = re.compile(
    r"\b(lost (money|savings|customers|clients|my job)|"
    r"wasted (money|time|months|years)|"
    r"burned (through|out)|going under|about to shut|failing|"
    r"out of (runway|money|cash|budget)|"
    r"invest(ed|ing) a lot|put in everything|"
    r"stressful|stressed|overwhelmed|anxious about|worried about|scared)\b",
    re.IGNORECASE,
)

# Matches full URLs AND bare domains (e.g. "simplifiedstartup.com/start-project").
# Used to deterministically strip any link the LLM emits in close/cta/post_cta/success/soft_close —
# the frontend buttons are the ONLY source of links, so any URL in the text is noise or a hallucination.
_URL_RE = re.compile(
    r"https?://\S+"
    r"|www\.\S+"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|app|io|co|org|net|ai)(?:/\S*)?",
    re.IGNORECASE,
)


def is_buying_signal(text: str, cta_offered: int = 0) -> bool:
    t = text or ""
    if _STRONG_BUYING_RE.search(t):
        return True
    if cta_offered > 0 and _WEAK_AFFIRM_RE.search(t):
        return True
    return False


def needs_empathy(text: str) -> bool:
    """Detect practical hardship (financial loss, stress) that warrants a one-liner of empathy."""
    return bool(_EMPATHY_RE.search(text or ""))


def _strip_urls(text: str) -> str:
    """Remove all URLs from text. Used in close phase so the LLM cannot
    inject a wrong link -- the frontend CTA card handles the booking link."""
    text = _URL_RE.sub("", text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([.,:])', r'\1', text)
    return text.strip(": ").strip()


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
- ready_for_cta: true ONLY when ALL FOUR conditions are met:
  (1) the founder's business stage is clearly stated,
  (2) their specific bottleneck is stated in their own words -- "inconsistent sales" or "no leads" counts,
      but a vague opener like "struggling to grow" does NOT count on its own,
  (3) you know at least one concrete detail: what they have already tried, how long the problem has existed,
      what their current marketing/sales/ops setup looks like, or how urgent this is for them,
  (4) at least 3 substantive exchanges have happened (not counting greetings).
  If ANY of these four is missing or vague, set this false. When in doubt, set false.
- should_offer_call_now: your CONSIDERED JUDGMENT about whether RIGHT NOW is the natural moment to offer the
  free strategy call. Think like a real advisor: offer the call when you genuinely understand their situation
  well enough that a real person could help, AND the founder has opened up enough that the offer feels earned.
  IMPORTANT: only count turns where the founder shared CONCRETE BUSINESS INFORMATION (stage, bottleneck, what
  they tried, their setup). Do NOT count greetings, meta questions about the bot, emotional statements alone,
  or one-line vague answers. You need at least 3 such substantive exchanges before setting this true.
  Set true when: you know their stage, real bottleneck, and at least one concrete detail, AND they have had
  at least 3 substantive exchanges sharing business information.
  Set false when: fewer than 3 substantive business exchanges have happened, they just gave a one-liner,
  they are mid-thought, or offering would interrupt a useful discovery thread.
  When genuinely unsure, set false and keep listening.
- call_offer_reasoning: one short clause explaining your should_offer_call_now decision.
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
    "- On pricing: if the founder directly asks what something costs or what the packages are, share the\n"
    "  KB-approved prices from the reference information provided. Only quote figures that appear in the\n"
    "  reference information -- never invent or estimate a number. If no pricing is in the reference\n"
    "  information, point them to the pricing page: " + PRICING_URL + "\n"
    "- On services: ONLY list services when the founder explicitly asks 'what services do you have' or 'what do\n"
    "  you offer' -- their EXACT names, never paraphrased: Digital Marketing, Website Development, Branding and\n"
    "  Growth, Sales and Lead Generation, AI Automation, Business and Startup Advisory, Talent and Staffing,\n"
    "  Bookkeeping and Accounting. When the founder asks 'which service should I start with' or 'what would\n"
    "  help me most' -- that is NOT a request to list services. Ask ONE clarifying question to narrow it down.\n"
    "  Never mention 'the AI Advisor', 'our advisor tool', or suggest the founder use the chatbot -- they are\n"
    "  already in it.\n"
    "- You never abbreviate the company name. Always write 'Simplified Startup' in full. Never write 'SS'.\n"
    "- You never use these words: leverage, synergistic, best-in-class, move the needle, holistically, unlock.\n\n"
    "STYLE:\n"
    "- Maximum 3 sentences. Warm and natural, never salesy or repetitive.\n"
    "- At most one question per message.\n"
    "- Never restart the conversation, never re-introduce yourself, and never repeat a point you already made.\n"
    "  Build on what the founder has told you and refer back to it in their own words.\n"
    "- Timeline questions: SEO usually takes 3 to 6 months for meaningful movement; paid ads and a new website\n"
    "  can move things in weeks. Say this plainly.\n"
    "- If asked whether you are human: say you are Simplified Startup's AI advisor and a real person can take it further.\n"
    "- If asked what Simplified Startup is, who owns/runs it, or anything about the company itself: give a brief, grounded\n"
    "  answer from the reference information provided (1-2 sentences), then pivot with one question about the founder's\n"
    "  business. Never fire CTA or offer the call just because someone asked about Simplified Startup.\n"
    "- Simplified Startup was founded by Thomas Meyers. If asked who owns or founded Simplified Startup, say Thomas Meyers\n"
    "  directly. Do not defer this to the strategy call.\n"
    "- Never say 'I don't know who owns' or 'I can't answer that' — if the reference information doesn't say, say\n"
    "  'a real person on the strategy call can answer that directly' and pivot to their business."
)

_EMPATHY_RIDER = (
    "\n\nEMPATHY NOTE: The founder just mentioned financial loss, burnout, or real stress. Start your reply "
    "with ONE short sentence that acknowledges what they are going through in human terms before anything "
    "else. Do not be clinical. Do not say 'I understand'. Say something a real person would say to a friend "
    "in that spot. Then continue with the rest of your reply."
)

_PHASE_DISCOVER = (
    "CURRENT GOAL -- GET TO KNOW THEM (do not sell yet):\n"
    "Do NOT offer, mention, or hint at the strategy call, and do NOT include any link.\n"
    "React briefly and warmly to what they just said, then ask ONE natural, specific question that moves you toward\n"
    "understanding their stage, their biggest bottleneck, or what they have already tried.\n\n"
    "DISCOVERY QUESTIONS MUST BE ABOUT THEIR BUSINESS SITUATION ONLY:\n"
    "- Good questions: What stage is the business at? What is the biggest thing holding growth back?\n"
    "  What have you already tried? How urgent is this? What does your current setup look like?\n"
    "- NEVER ask about: what features they want built, what content they want to create, what platform\n"
    "  they want to use, what their website should look like, what their logo should be, or any other\n"
    "  specific tactical detail. You are not scoping the project. You are learning the situation.\n\n"
    "CHIPS (2 to 4 tappable answer options) must each be a VALID ANSWER to the question you just asked, in the\n"
    "founder's own voice. Every chip MUST:\n"
    "  - Be a FULL first-person sentence starting with 'I' or 'We' (never a label, never a tag).\n"
    "  - Be reflective and specific, 8 to 16 words. Sound like how a founder would actually describe their\n"
    "    situation to a friend, not like a VC pitch deck or a dropdown option.\n"
    "  - BAD examples (do not emit these): 'Idea / pre-seed', 'Seed / early traction', 'Growth / scaling',\n"
    "    'MVP stage', 'Pre-revenue'. These are category labels, not answers.\n"
    "  - GOOD examples for a stage question: 'I'm still figuring out what to build and who it's for',\n"
    "    'I'm building the product but haven't launched it to anyone yet',\n"
    "    'I've got a handful of early users and I'm trying to grow',\n"
    "    'The business is running and I want to scale it faster'.\n"
    "  - GOOD examples for a bottleneck question: 'I can't get people to actually try what I built',\n"
    "    'My marketing is running but it is not bringing in leads'.\n"
    "  - GOOD examples for an 'already tried' question: 'I ran some ads but nothing stuck',\n"
    "    'I posted on social for a few weeks and gave up'.\n"
    "Never name a competitor product -- not even as an example of what the founder tried.\n"
    "  Never name Wix, WordPress, Squarespace, Shopify, Webflow, Framer, Fiverr, Upwork, or any other.\n"
    "  Describe the situation or outcome instead -- 'I tried a website builder but it did not match my brand'\n"
    "  not 'I tried Wix'. 'I hired someone but the project stalled' not 'I hired a freelancer on Fiverr'.\n"
    "Do NOT add a 'Something else' chip -- that is handled by the frontend automatically."
)

_PHASE_CTA = (
    "CURRENT GOAL -- OFFER THE CALL, ONCE:\n"
    "You now understand their situation. In one or two sentences, reflect back the specific thing they care about\n"
    "using their own words, then warmly offer a free 30-minute strategy call with a real person from Simplified\n"
    "Startup. Include this exact link at the end: " + BOOK_URL + "\n"
    "Do NOT ask for a meeting time. Do NOT ask for their email. Just make the offer feel natural and share the link."
)

_PHASE_SOFT_CLOSE = (
    "CURRENT GOAL -- THEY ARE UNSURE, GIVE THEM TWO LOW-PRESSURE PATHS:\n"
    "The founder has signalled they do not know what to tackle, or they are not ready to click a booking link yet.\n"
    "Reply in two short sentences:\n"
    "1. Acknowledge warmly that it is okay to not have it all mapped out yet.\n"
    "2. Say they can either explore what Simplified Startup actually does on the website, OR hop on a free\n"
    "   30-minute strategy call with a real person who will help them figure out the next step.\n"
    "Do NOT include any URL in the text -- the booking link and the explore link are added by the system as buttons.\n"
    "Do NOT push. Do NOT ask for their email. Do NOT ask a question at the end."
)

_PHASE_OBJECTION = (
    "CURRENT GOAL -- ANSWER THEM STRAIGHT:\n"
    "You have already offered the call once. Answer their question or concern honestly and briefly, using only what\n"
    "you actually know about Simplified Startup. Do not give advice or how-to.\n"
    "On pricing: if they ask what something costs, share the KB-approved prices from the reference information\n"
    "provided -- only quote figures that actually appear in the reference information, never estimate or invent.\n"
    "If pricing is in the reference information, quote the specific figure AND include the pricing page link so\n"
    "they can see the full breakdown: " + PRICING_URL + "\n"
    "If no pricing is in the reference information, point them to: " + PRICING_URL + "\n"
    "On trust / past bad experience: acknowledge what happened in one short sentence before answering. Do not\n"
    "lecture, do not defend, do not oversell.\n"
    "If it genuinely fits, you may offer the call one more time with this link: " + BOOK_URL + " -- but do not push,\n"
    "do not repeat yourself, and never ask for a meeting time or their email."
)

_PHASE_CLOSE = (
    "CURRENT GOAL -- THEY'RE READY, HAND THEM THE LINK:\n"
    "The founder has signalled they want to take the next step. Reply in exactly this shape and nothing else:\n"
    "one short warm sentence acknowledging it, then this exact sentence and nothing after it:\n"
    "'Book your free 30-minute strategy call — you'll leave with a written plan either way.'\n"
    "Do NOT include any URL or link in your reply -- the booking link is added automatically by the system.\n"
    "Do NOT greet them, do NOT re-introduce yourself, do NOT ask a question, do NOT ask for their email or a meeting\n"
    "time, and do NOT invent any other process (no 'we'll send you a draft', no 'within a week', no forms)."
)

_PHASE_SUCCESS = (
    "CURRENT GOAL -- POINT THEM TO PROOF:\n"
    "The founder asked for case studies, past results, or how good the service actually is.\n"
    "Reply in two short sentences:\n"
    "1. Say briefly that Simplified Startup has helped founders across different stages -- without inventing\n"
    "   specific numbers, names, or outcomes.\n"
    "2. Point them to the success stories page for the real examples, and offer a quick call if they want to\n"
    "   talk through whether it fits their situation.\n"
    "Do NOT include any URL in the text -- the success-stories link and the booking link are added by the system\n"
    "as buttons. Do NOT invent client names, revenue numbers, or percentages."
)

_PHASE_POST_CTA = (
    "CURRENT GOAL -- KEEP THE CONVERSATION GOING, NO CALL PUSH:\n"
    "You have already offered the strategy call. Do NOT mention it again, do NOT include any link,\n"
    "and do NOT ask them to book or schedule anything.\n"
    "The founder is still in the conversation -- they may have more questions, concerns, or things to share.\n"
    "React warmly and naturally to what they just said, exactly as you would in discover phase.\n"
    "Put your reply in the 'reply' field. Ask ONE genuine follow-up question based on what they told you,\n"
    "or answer their question plainly using only what you know about Simplified Startup.\n"
    "If they ask about the call specifically (what happens, is it free, do they have to pay),\n"
    "answer that question directly and naturally -- but only if they asked.\n"
    "Never push, never repeat the offer, never add a closing line about taking the next step.\n\n"
    "DIRECT RECOMMENDATION RULE:\n"
    "If the founder explicitly asks you to recommend between two options they named -- e.g. 'which should I\n"
    "hire first', 'which of these two would you go with', 'what do you think I should focus on first' --\n"
    "give ONE short grounded steer in 1-2 sentences, based ONLY on what they told you in this conversation.\n"
    "Do NOT explain how to execute either option. Do NOT give steps or a plan. Just say which makes more\n"
    "sense given their specific situation and why in one line. End with a nudge that the strategy call will\n"
    "help map it out properly. Example: 'Based on what you've shared -- solo founder, product still being\n"
    "built -- a junior dev is usually the more immediate unlock. A PM makes more sense once there's something\n"
    "to manage. The strategy call can help you think through this for your specific setup.'\n"
    "This is the ONLY situation where you give a directional opinion. For everything else, keep discovering.\n\n"
    "CHIPS: provide 2 to 4 tappable options in the 'chips' field that are natural ways the founder might\n"
    "respond to your reply -- full first-person sentences, 8 to 16 words, in their voice (same style as discovery\n"
    "chips). If your reply is answering a factual question rather than asking one, chips can be natural\n"
    "follow-up things they might want to know next. Never put a link or a 'book the call' option in a chip.\n"
    "Do NOT add a 'Something else' chip -- the frontend adds that automatically."
)

_PHASES = {
    "discover": _PHASE_DISCOVER,
    "cta": _PHASE_CTA,
    "soft_close": _PHASE_SOFT_CLOSE,
    "objection": _PHASE_OBJECTION,
    "post_cta": _PHASE_POST_CTA,
    "close": _PHASE_CLOSE,
    "success": _PHASE_SUCCESS,
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


def _last_bot_question(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            text = m.get("content", "")
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            for s in reversed(sentences):
                if s.endswith("?"):
                    return s
            return ""
    return ""


def _user_content(profile: FounderProfile, kb_text: str, transcript: str, messages: list[dict] = None) -> str:
    profile_text = _profile_summary(profile)
    last_q = _last_bot_question(messages or [])
    context_block = f"WHAT YOU KNOW ABOUT THIS FOUNDER SO FAR: {profile_text}"
    if last_q:
        context_block += f"\nTHE LAST QUESTION YOU ASKED THEM: {last_q}"
        context_block += "\nIMPORTANT: Your reply must directly respond to that question using what the founder just said. Never restart with a generic opener. Never ignore what they told you."
    # Trim KB text aggressively for latency when it is huge; keep most relevant chunks only.
    if kb_text and len(kb_text) > 4000:
        kb_text = kb_text[:4000] + "\n[...kb truncated for latency...]"
    return (
        f"{context_block}\n\n"
        f"REFERENCE INFORMATION about Simplified Startup (quote prices only from this text, never invent figures):\n"
        f"{kb_text}\n\n"
        f"CONVERSATION SO FAR:\n{transcript}"
    )


class _ChipsOnly(BaseModel):
    chips: List[str] = Field(default_factory=list)


_CHIPS_ONLY_SYSTEM = (
    "You generate 2 to 4 tappable reply chips for a founder talking to the Simplified Startup AI Advisor.\n"
    "Given the advisor's latest message, produce natural ways the founder might respond -- each a full\n"
    "first-person sentence, 8 to 16 words, in the founder's voice. If the advisor asked a question, the chips\n"
    "answer it. If the advisor answered something, the chips are natural next things the founder might say or ask.\n"
    "Never include a link or a 'book a call' option. Return only the chips."
)


def _generate_chips_for(reply_text: str, transcript: str) -> list:
    """Cheap chip-only call used when the main structured call returns no chips."""
    if not llm.settings.has_llm:
        return []
    try:
        out = llm.structured(
            system=_CHIPS_ONLY_SYSTEM,
            user=f"CONVERSATION SO FAR:\n{transcript}\n\nADVISOR'S LATEST MESSAGE:\n{reply_text}",
            response_model=_ChipsOnly,
            model=llm.settings.fast_model,
            temperature=0.4,
        )
        return [c.strip() for c in (out.chips or []) if c and c.strip()][:6]
    except Exception:
        return []


def generate_reply(messages: list[dict], kb_chunks: list[str], phase: str, profile: FounderProfile):
    """Returns (reply_text, chips_list). chips are only produced in the discover phase."""
    if not llm.settings.has_llm:
        return ("Tell me a bit about what you're building and where you're stuck.", [])

    transcript = llm.transcript_text(messages)
    kb_text = "\n\n".join(kb_chunks) if kb_chunks else ""
    user = _user_content(profile, kb_text, transcript, messages)

    # Inject empathy rider if the latest user message signals practical hardship.
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break
    persona = _BASE_PERSONA
    if needs_empathy(last_user_msg):
        persona = persona + _EMPATHY_RIDER

    # discover and post_cta both produce tappable chips alongside the reply.
    if phase in ("discover", "post_cta"):
        system = persona + "\n\n" + _PHASES.get(phase, _PHASE_DISCOVER)
        try:
            out = llm.structured(
                system=system,
                user=user,
                response_model=DiscoverTurn,
                temperature=0.4,
            )
            chips = [c.strip() for c in (out.chips or []) if c and c.strip()][:6]
            reply = (out.reply or "").strip()
            if phase == "post_cta":
                reply = _strip_urls(reply)
            if reply:
                # If the model gave a reply but no chips, make a cheap chip-only
                # follow-up call so post_cta/discover never render chip-less.
                if not chips:
                    chips = _generate_chips_for(reply, transcript)
                return (reply, chips)
        except Exception:
            pass
        text = llm.generate(system, user, temperature=0.4)
        if phase == "post_cta":
            text = _strip_urls(text)
        chips = _generate_chips_for(text, transcript)
        return (text, chips)

    system = persona + "\n\n" + _PHASES.get(phase, _PHASE_DISCOVER)
    text = llm.generate(system, user, temperature=0.4)

    # Strip ALL URLs deterministically on every phase except objection — the frontend buttons are the
    # only source of links, so any URL the LLM emits is either redundant or a hallucination.
    # Objection phase is the one exception: it may legitimately cite the pricing page URL.
    if phase != "objection":
        text = _strip_urls(text)

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