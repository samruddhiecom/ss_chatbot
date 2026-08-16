"""The staged conversation as a LangGraph state machine.

One invocation processes one founder turn:
    classify intent -> (bounded refusal | run stage) -> output grounding check

The guard sandwich is explicit here: the input rail runs first (classify), the
retrieval + generation happen inside the stage node, and the output rail runs last.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from config import settings
from app import llm, stages, voice
from app.rails import input_rail, output_rail
from app.schemas import (
    NON_PROGRESSING_INTENTS,
    GraphState,
    Intent,
    Stage,
    next_stage,
)
from app.stages import STAGE_LABEL


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def classify_intent(state: GraphState) -> dict:
    intent = input_rail.classify(state.get("messages", []), state.get("last_user_input", ""))
    return {"intent": intent.value}


def handle_refusal(state: GraphState) -> dict:
    intent = Intent(state["intent"])
    stage = state["current_stage"]
    label = STAGE_LABEL.get(stage, "plan")
    reply, gap = input_rail.bounded_response(intent, label)

    gaps = list(state.get("named_gaps", []))
    if gap and gap not in gaps:
        gaps.append(gap)

    messages = list(state.get("messages", [])) + [{"role": "assistant", "content": reply}]
    return {"assistant_reply": reply, "messages": messages, "named_gaps": gaps, "snapshot_ready": False}


def run_stage(state: GraphState) -> dict:
    stage = state["current_stage"]
    messages = list(state.get("messages", []))
    followups = dict(state.get("stage_followups", {}))
    captured = dict(state.get("captured", {}))

    ext = stages.extract(stage, messages)
    if ext.captured:
        captured[stage] = {**captured.get(stage, {}), **ext.captured}

    used = followups.get(stage, 0)
    advance = ext.covered or used >= settings.max_followups_per_stage

    snapshot_ready = False
    new_stage = stage

    if advance:
        nxt = next_stage(Stage(stage))
        if nxt == Stage.DONE:
            reply = voice.apply_voice("That's enough for me to put your snapshot together.")
            new_stage = Stage.DONE.value
            snapshot_ready = True
        else:
            bridge = stages.acknowledge_and_advance(stage, STAGE_LABEL[nxt.value])
            reply = f"{bridge} {stages.opener(nxt.value)}".strip()
            new_stage = nxt.value
    else:
        followups[stage] = used + 1
        reply = stages.styled_followup(ext.follow_up)

    messages.append({"role": "assistant", "content": reply})
    return {
        "assistant_reply": reply,
        "messages": messages,
        "current_stage": new_stage,
        "stage_followups": followups,
        "captured": captured,
        "snapshot_ready": snapshot_ready,
    }


def output_check(state: GraphState) -> dict:
    reply = state.get("assistant_reply", "")
    sources = llm.transcript_text(state.get("messages", []))
    _clean, flags = output_rail.verify_turn(reply, sources)
    existing = list(state.get("grounding_flags", []))
    return {"grounding_flags": existing + flags}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def route_after_intent(state: GraphState) -> str:
    return "refusal" if Intent(state["intent"]) in NON_PROGRESSING_INTENTS else "stage"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("classify", classify_intent)
    builder.add_node("refusal", handle_refusal)
    builder.add_node("stage", run_stage)
    builder.add_node("check", output_check)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_intent, {"refusal": "refusal", "stage": "stage"})
    builder.add_edge("refusal", "check")
    builder.add_edge("stage", "check")
    builder.add_edge("check", END)
    return builder.compile()


GRAPH = build_graph()
