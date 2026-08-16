"""FastAPI engine. This is the backend Himanshu's embed shell calls.

Sessions are in-memory for the draft build; production swaps this for the shared
store with 30-day transcript retention and per-visitor + global rate limits.
"""
from __future__ import annotations

import threading
import uuid
from datetime import date

from fastapi import FastAPI, HTTPException

from config import settings
from app import capture as capture_mod
from app import snapshot as snapshot_mod
from app.graph import GRAPH
from app.schemas import (
    CaptureRequest,
    CaptureResponse,
    MessageRequest,
    MessageResponse,
    SessionRequest,
    SnapshotResponse,
    Stage,
    StartResponse,
)
from app.stages import opener

app = FastAPI(title="SS Business Planning Chatbot — Engine", version="0.1.0-draft")

DISCLOSURE = [
    "Automated planning assistant.",
    "Conversations may be reviewed.",
    "General guidance, not professional advice.",
]

# --------------------------------------------------------------------------- #
# In-memory session + rate-limit state (draft only)
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()
_SESSIONS: dict[str, dict] = {}
_SNAPSHOTS: dict[str, object] = {}
_GLOBAL = {"day": date.today().isoformat(), "count": 0}


def _check_global_cap() -> None:
    today = date.today().isoformat()
    if _GLOBAL["day"] != today:
        _GLOBAL["day"] = today
        _GLOBAL["count"] = 0
    if _GLOBAL["count"] >= settings.global_daily_cap:
        raise HTTPException(status_code=429, detail="Daily capacity reached. Please try again tomorrow.")
    _GLOBAL["count"] += 1


def _get_session(session_id: str) -> dict:
    state = _SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. Call /conversation/start first.")
    return state


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_configured": settings.has_llm, "model": settings.smart_model}


@app.post("/conversation/start", response_model=StartResponse)
def start() -> StartResponse:
    with _LOCK:
        _check_global_cap()
        sid = uuid.uuid4().hex
        first = opener(Stage.IDEA.value)
        _SESSIONS[sid] = {
            "session_id": sid,
            "messages": [{"role": "assistant", "content": first}],
            "current_stage": Stage.IDEA.value,
            "stage_followups": {},
            "captured": {},
            "named_gaps": [],
            "grounding_flags": [],
            "snapshot_ready": False,
        }
    return StartResponse(session_id=sid, disclosure=DISCLOSURE, message=first, stage=Stage.IDEA.value)


@app.post("/conversation/message", response_model=MessageResponse)
def message(req: MessageRequest) -> MessageResponse:
    with _LOCK:
        _check_global_cap()
        state = _get_session(req.session_id)

        user_turns = sum(1 for m in state["messages"] if m["role"] == "user")
        if user_turns >= settings.max_messages_per_session:
            raise HTTPException(status_code=429, detail="This session has reached its message limit.")

        state["messages"].append({"role": "user", "content": req.message})
        state["last_user_input"] = req.message

    # Graph invocation outside the lock (it may make network calls).
    result = GRAPH.invoke(state)

    with _LOCK:
        _SESSIONS[req.session_id] = result

    return MessageResponse(
        message=result.get("assistant_reply", ""),
        stage=result.get("current_stage", Stage.IDEA.value),
        intent=result.get("intent", ""),
        snapshot_ready=bool(result.get("snapshot_ready", False)),
        grounding_flags=result.get("grounding_flags", []),
    )


@app.post("/conversation/snapshot", response_model=SnapshotResponse)
def get_snapshot(req: SessionRequest) -> SnapshotResponse:
    state = _get_session(req.session_id)
    snap, report = snapshot_mod.assemble(state["messages"], state.get("named_gaps", []))
    with _LOCK:
        _SNAPSHOTS[req.session_id] = snap
    return SnapshotResponse(snapshot=snap, grounding_report=report)


@app.post("/capture", response_model=CaptureResponse)
def capture(req: CaptureRequest) -> CaptureResponse:
    _get_session(req.session_id)
    snap = _SNAPSHOTS.get(req.session_id)
    rec = capture_mod.capture_lead(req.session_id, req.name, req.email, snap)
    return CaptureResponse(
        captured=True,
        tool_source=rec["tool_source"],
        snapshot_attached=rec["snapshot_attached"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
