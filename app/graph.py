"""LangGraph state machine for the SS AI Advisor.

Flow per user turn:
  classify intent → (refusal | advisor) → output check

The advisor node:
  - extracts founder profile
  - if not covered: asks one follow-up
  - if covered: gives one useful insight + natural CTA to book a call
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


def run_advisor(state: GraphState) -> dict:
    messages = list(state.get("messages", []))
    cta_count = state.get("cta_offered", 0) or 0
    profile_dict = state.get("founder_profile", {})

    from app.schemas import FounderProfile
    profile = FounderProfile(**profile_dict) if profile_dict else advisor.extract_profile(messages)

    # If the founder has sent at least one message, mark as covered — enough to pull to call
    user_messages = [m for m in messages if m.get("role") == "user"]
    if user_messages and not profile.covered:
        profile.covered = True

    from app.kb import store
    kb_chunks = []
    if profile.bottleneck:
        kb_chunks = store.query(profile.bottleneck, top_k=5)
    elif profile.service_interest:
        kb_chunks = store.query(profile.service_interest, top_k=5)

    # Once profile is covered, next reply includes the CTA
    cta_ready = bool(len([m for m in messages if m.get("role") == "user"]) >= 1)
    if cta_ready:
        cta_count += 1

    reply = advisor.generate_reply(messages, kb_chunks, cta_count)

    messages.append({"role": "assistant", "content": reply})

    return {
        "assistant_reply": reply,
        "messages": messages,
        "founder_profile": profile.model_dump(),
        "cta_ready": cta_ready,
        "cta_offered": cta_count,
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
