"""LangGraph state machine for the SS AI Advisor.

Flow per user turn:
  classify intent -> (refusal | advisor) -> output check

The advisor node runs a phased conversation:
  discover  -> learn the founder's situation (reply + tappable chips), no call yet
  cta       -> once understood, offer the free call once, with the booking link
  objection -> answer their concern, may re-offer the call once
  close     -> they signalled readiness, hand over the booking link now
"""
from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph

from app import advisor, llm
from app.rails import input_rail, output_rail
from app.schemas import (
    NON_PROGRESSING_INTENTS,
    GraphState,
    Intent,
)

MAX_DISCOVERY_TURNS = 6
MIN_DISCOVERY_TURNS = 2

_SERVICE_Q_RE = re.compile(
    r"\b(what services|what do you offer|what can you help|what do you do|"
    r"what areas|what kind of (work|help|services)|services (do you|you) (offer|provide|have)|"
    r"tell me (about|more about) your services|list (your|the) services)\b",
    re.IGNORECASE,
)
_PRICING_Q_RE = re.compile(
    r"\b(how much|what('s| is) (the |your )?(cost|price|pricing|rate)|"
    r"pricing|price|cost|rates|packages|plans|fees|monthly fee|retainer|"
    r"what do you charge|how are you priced)\b",
    re.IGNORECASE,
)


def _detect_kb_route(text: str):
    if _PRICING_Q_RE.search(text or ""):
        return "pricing"
    if _SERVICE_Q_RE.search(text or ""):
        return "service"
    return None


def classify_intent(state: GraphState) -> dict:
    intent = input_rail.classify(
        state.get("messages", []),
        state.get("last_user_input", "")
    )
    return {"intent": intent.value}


def handle_refusal(state: GraphState) -> dict:
    intent = Intent(state["intent"])
    if intent == Intent.EXISTING_CLIENT:
        reply = (
            "This one\u2019s better handled by a person. "
            "Reach your named contact directly, or email simplifiedstartupllc@gmail.com "
            "and someone will get back to you."
        )
    else:
        reply, _ = input_rail.bounded_response(intent, "your situation")
    messages = list(state.get("messages", [])) + [{"role": "assistant", "content": reply}]
    return {
        "assistant_reply": reply,
        "messages": messages,
        "chips": [],
        "cta_ready": False,
    }


def _build_retrieval_query(profile, last_user_input: str) -> str:
    parts = []
    if last_user_input:
        parts.append(last_user_input.strip())
    if profile.bottleneck and profile.bottleneck not in (last_user_input or ""):
        parts.append(profile.bottleneck)
    if profile.service_interest and profile.service_interest not in " ".join(parts):
        parts.append(profile.service_interest)
    return " ".join(parts) if parts else (profile.bottleneck or "startup advisory")


def _substantive_user_turns(messages) -> int:
    count = 0
    for m in messages:
        if m.get("role") != "user":
            continue
        if input_rail.looks_like_greeting(m.get("content", "")):
            continue
        count += 1
    return count


def _decide_phase(profile, last_user_input: str, user_turns: int, cta_offered: int) -> str:
    if advisor.is_buying_signal(last_user_input, cta_offered):
        return "close"
    ready = bool(getattr(profile, "ready_for_cta", False))
    discovery_done = (ready and user_turns >= MIN_DISCOVERY_TURNS) or user_turns >= MAX_DISCOVERY_TURNS
    if cta_offered == 0 and not discovery_done:
        return "discover"
    if cta_offered == 0 and discovery_done:
        return "cta"
    return "objection"


def run_advisor(state: GraphState) -> dict:
    messages = list(state.get("messages", []))
    cta_offered = int(state.get("cta_offered", 0) or 0)
    last_user_input = state.get("last_user_input", "")
    user_turns = _substantive_user_turns(messages)

    profile = advisor.extract_profile(messages)
    phase = _decide_phase(profile, last_user_input, user_turns, cta_offered)

    from app.kb import store

    kb_route = _detect_kb_route(last_user_input)
    if kb_route:
        typed_chunks = store.query_by_type(kb_route, top_k=6)
        retrieval_query = _build_retrieval_query(profile, last_user_input)
        cosine_chunks = store.query(retrieval_query, top_k=4)
        seen = set()
        kb_chunks = []
        for c in typed_chunks + cosine_chunks:
            if c not in seen:
                seen.add(c)
                kb_chunks.append(c)
    else:
        retrieval_query = _build_retrieval_query(profile, last_user_input)
        kb_chunks = store.query(retrieval_query, top_k=10) if retrieval_query else []

    reply, chips = advisor.generate_reply(messages, kb_chunks, phase, profile)

    # Store the kb_text for use in the output rail regeneration closure
    kb_text = "\n\n".join(kb_chunks) if kb_chunks else ""

    # Output rail with regeneration — runs before appending to messages
    sources = llm.transcript_text(messages)

    def _regen(correction_prompt: str) -> str:
        """Regenerate the reply with a correction prompt using the same phase/profile context."""
        system = (
            "You are the Simplified Startup AI Advisor. "
            "Rewrite the reply below fixing only the flagged issues. "
            "Keep the same intent, phase, and conversational tone.\n\n"
            + correction_prompt
        )
        return llm.generate(system, f"KB context:\n{kb_text}\n\nTranscript:\n{sources}", temperature=0.3)

    reply, flags = output_rail.regenerate_if_flagged(reply, sources, _regen)

    messages.append({"role": "assistant", "content": reply})

    offered = phase in ("cta", "close")
    new_cta_offered = cta_offered + (1 if offered else 0)

    return {
        "assistant_reply": reply,
        "messages": messages,
        "chips": chips,
        "founder_profile": profile.model_dump(),
        "conv_stage": phase,
        "cta_ready": offered,
        "cta_offered": new_cta_offered,
        "grounding_flags": flags,
    }


def output_check(state: GraphState) -> dict:
    # Flags already computed in run_advisor during regeneration.
    # This node is kept for future extensibility (e.g. LLM claims check).
    existing = list(state.get("grounding_flags", []))
    return {"grounding_flags": existing}


def route_after_intent(state: GraphState) -> str:
    return "refusal" if Intent(state["intent"]) in NON_PROGRESSING_INTENTS else "advisor"


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("classify", classify_intent)
    builder.add_node("refusal", handle_refusal)
    builder.add_node("advisor", run_advisor)
    builder.add_node("check", output_check)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify", route_after_intent,
        {"refusal": "refusal", "advisor": "advisor"}
    )
    builder.add_edge("refusal", "check")
    builder.add_edge("advisor", "check")
    builder.add_edge("check", END)
    return builder.compile()


GRAPH = build_graph()