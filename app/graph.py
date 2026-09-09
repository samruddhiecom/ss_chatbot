"""LangGraph state machine for the SS AI Advisor.

Flow per user turn:
  classify intent → (refusal | advisor) → output check

The advisor node:
  - extracts founder profile
  - if not covered: asks one follow-up
  - if covered: generates recommendation + CTA
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app import advisor, llm
from app.rails import input_rail, output_rail
from app.schemas import (
    NON_PROGRESSING_INTENTS,
    ConvStage,
    GraphState,
    Intent,
)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def classify_intent(state: GraphState) -> dict:
    intent = input_rail.classify(
        state.get("messages", []),
        state.get("last_user_input", "")
    )
    return {"intent": intent.value}


def handle_refusal(state: GraphState) -> dict:
    intent = Intent(state["intent"])
    # Existing client → escalate immediately (KB Section 14)
    if intent == Intent.EXISTING_CLIENT:
        reply = (
            "This one's better handled by a person. "
            "Reach your named contact directly, or email simplifiedstartupllc@gmail.com "
            "and someone will get back to you."
        )
    else:
        reply, _ = input_rail.bounded_response(intent, "your situation")
    messages = list(state.get("messages", [])) + [{"role": "assistant", "content": reply}]
    return {
        "assistant_reply": reply,
        "messages": messages,
        "recommendation_ready": False,
    }


def run_advisor(state: GraphState) -> dict:
    messages = list(state.get("messages", []))
    cta_count = state.get("cta_offered", 0)
    profile_dict = state.get("founder_profile", {})

    # Extract profile from full conversation
    from app.schemas import FounderProfile
    profile = FounderProfile(**profile_dict) if profile_dict else advisor.extract_profile(messages)

    recommendation_ready = False
    recommendation = state.get("recommendation")

    if profile.covered and not recommendation:
        # Build recommendation
        rec = advisor.build_recommendation(messages, profile)
        recommendation = rec.model_dump()
        recommendation_ready = True
        # Count CTA
        if "strategy call" in rec.cta.lower() or "book" in rec.cta.lower():
            cta_count = (cta_count or 0) + 1

    # Generate conversational reply
    kb_chunks = []
    if profile.bottleneck:
        from app.kb import store
        kb_chunks = store.query(profile.bottleneck, top_k=5)

    reply = advisor.generate_reply(messages, kb_chunks, cta_count or 0)

    # If recommendation ready, append it to the reply
    if recommendation_ready and recommendation:
        rec_obj = recommendation
        service_line = f"\n\n**My recommendation: {rec_obj['primary_service']}**"
        if rec_obj.get("secondary_service"):
            service_line += f" + {rec_obj['secondary_service']}"
        link = rec_obj.get("page_link", "")
        link_line = f"\n\nLearn more: simplified-startup-ui.vercel.app{link}" if link else ""
        reply = reply + service_line + link_line

    messages.append({"role": "assistant", "content": reply})

    return {
        "assistant_reply": reply,
        "messages": messages,
        "founder_profile": profile.model_dump(),
        "recommendation": recommendation,
        "recommendation_ready": recommendation_ready,
        "cta_offered": cta_count,
    }


def output_check(state: GraphState) -> dict:
    reply = state.get("assistant_reply", "")
    sources = llm.transcript_text(state.get("messages", []))
    _, flags = output_rail.verify_turn(reply, sources)
    existing = list(state.get("grounding_flags", []))
    return {"grounding_flags": existing + flags}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def route_after_intent(state: GraphState) -> str:
    return "refusal" if Intent(state["intent"]) in NON_PROGRESSING_INTENTS else "advisor"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("classify", classify_intent)
    builder.add_node("refusal", handle_refusal)
    builder.add_node("advisor", run_advisor)
    builder.add_node("check", output_check)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_after_intent,
        {"refusal": "refusal", "advisor": "advisor"}
    )
    builder.add_edge("refusal", "check")
    builder.add_edge("advisor", "check")
    builder.add_edge("check", END)
    return builder.compile()


GRAPH = build_graph()
