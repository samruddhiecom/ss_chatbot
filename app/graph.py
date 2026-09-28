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


def _build_retrieval_query(profile, last_user_input: str) -> str:
    """Build a richer retrieval query by combining the last user message with
    the extracted bottleneck and service interest.

    The original code used only profile.bottleneck or profile.service_interest
    (a 2-5 word extracted phrase), which produces a weak cosine match against
    the enriched 200-word chunks in Chroma. Including the full user message
    gives BGE-small more semantic signal to anchor on.
    """
    parts = []
    if last_user_input:
        parts.append(last_user_input.strip())
    if profile.bottleneck and profile.bottleneck not in (last_user_input or ""):
        parts.append(profile.bottleneck)
    if profile.service_interest and profile.service_interest not in " ".join(parts):
        parts.append(profile.service_interest)
    return " ".join(parts) if parts else (profile.bottleneck or "startup advisory")


def run_advisor(state: GraphState) -> dict:
    messages = list(state.get("messages", []))
    cta_count = state.get("cta_offered", 0) or 0
    profile_dict = state.get("founder_profile", {})

    from app.schemas import FounderProfile
    profile = FounderProfile(**profile_dict) if profile_dict else advisor.extract_profile(messages)

    user_messages = [m for m in messages if m.get("role") == "user"]
    if user_messages and not profile.covered:
        profile.covered = True

    from app.kb import store

    # Build expanded query — full last message + extracted signals
    last_user_input = state.get("last_user_input", "")
    retrieval_query = _build_retrieval_query(profile, last_user_input)

    kb_chunks = []
    if retrieval_query:
        # top_k=10 → reranker in store.query() prunes to settings.retrieval_top_k
        kb_chunks = store.query(retrieval_query, top_k=10, rerank=True)

    cta_ready = bool(len(user_messages) >= 1)
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
