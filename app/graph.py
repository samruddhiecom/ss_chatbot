"""LangGraph state machine for the SS AI Advisor.

Flow per user turn:
  classify intent -> (refusal | advisor) -> output check

The advisor node runs a phased conversation:
  discover  -> learn the founder's situation, no call offered yet
  cta       -> once understood, offer the free call once, with the booking link
  objection -> answer their concern, may re-offer the call once
  close     -> they signalled readiness, hand over the booking link now
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app import advisor, llm
from app.rails import input_rail, output_rail
from app.schemas import (
    NON_PROGRESSING_INTENTS,
    GraphState,
    Intent,
)


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
        "cta_ready": False,
    }


def _build_retrieval_query(profile, last_user_input: str) -> str:
    """Combine the last user message with the extracted bottleneck and service
    interest so BGE-small has enough semantic signal to match the KB chunks."""
    parts = []
    if last_user_input:
        parts.append(last_user_input.strip())
    if profile.bottleneck and profile.bottleneck not in (last_user_input or ""):
        parts.append(profile.bottleneck)
    if profile.service_interest and profile.service_interest not in " ".join(parts):
        parts.append(profile.service_interest)
    return " ".join(parts) if parts else (profile.bottleneck or "startup advisory")


def _decide_phase(profile, last_user_input: str, user_turns: int, cta_offered: int) -> str:
    """Pick the conversation phase for this turn.

    - A clear buying signal at any point jumps straight to close.
    - Otherwise we stay in discover until we understand the bottleneck (and have had
      at least two exchanges), or until three user turns have passed, then offer the
      call once. After the call has been offered, further turns are objection handling.
    """
    if advisor.is_buying_signal(last_user_input):
        return "close"
    discovery_done = (user_turns >= 2 and bool(profile.bottleneck)) or user_turns >= 3
    if cta_offered == 0 and not discovery_done:
        return "discover"
    if cta_offered == 0 and discovery_done:
        return "cta"
    return "objection"


def run_advisor(state: GraphState) -> dict:
    messages = list(state.get("messages", []))
    cta_offered = int(state.get("cta_offered", 0) or 0)
    last_user_input = state.get("last_user_input", "")
    user_turns = sum(1 for m in messages if m.get("role") == "user")

    # Re-read the founder from the whole conversation each turn.
    profile = advisor.extract_profile(messages)

    phase = _decide_phase(profile, last_user_input, user_turns, cta_offered)

    from app.kb import store
    retrieval_query = _build_retrieval_query(profile, last_user_input)
    kb_chunks = store.query(retrieval_query, top_k=10) if retrieval_query else []

    reply = advisor.generate_reply(messages, kb_chunks, phase, profile)
    messages.append({"role": "assistant", "content": reply})

    offered = phase in ("cta", "close")
    new_cta_offered = cta_offered + (1 if offered else 0)

    return {
        "assistant_reply": reply,
        "messages": messages,
        "founder_profile": profile.model_dump(),
        "conv_stage": phase,
        "cta_ready": offered,
        "cta_offered": new_cta_offered,
    }


def output_check(state: GraphState) -> dict:
    reply = state.get("assistant_reply", "")
    sources = llm.transcript_text(state.get("messages", []))
    _, flags = output_rail.verify_turn(reply, sources)
    existing = list(state.get("grounding_flags", []))
    return {"grounding_flags": existing + flags}


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