"""FastAPI engine for the SS AI Advisor.

Endpoints:
  POST /conversation/start    — disclosure + opening message
  POST /conversation/message  — send a message, get a reply with CTA when ready
  POST /capture               — submit lead (name + email)
  POST /webhook/advisor-1     — adapter for the ss-advisor frontend (translates
                                the frontend contract to the internal graph)
  GET  /health
"""
from __future__ import annotations

import threading
import uuid
from datetime import date

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from app import capture as capture_mod
from app.graph import GRAPH
from app.schemas import (
    CaptureRequest,
    CaptureResponse,
    MessageRequest,
    MessageResponse,
    StartResponse,
)
from app import advisor as adv

app = FastAPI(title="SS AI Advisor Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLOSURE = [
    "Automated planning assistant.",
    "Conversations may be reviewed.",
    "General guidance, not professional advice.",
]

BOOK_URL = "https://simplified-startup-ui.vercel.app/#book"

_LOCK = threading.Lock()
_SESSIONS: dict[str, dict] = {}
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_configured": settings.has_llm, "model": settings.smart_model}


@app.post("/conversation/start", response_model=StartResponse)
def start() -> StartResponse:
    with _LOCK:
        _check_global_cap()
        sid = uuid.uuid4().hex
        _SESSIONS[sid] = {
            "session_id": sid,
            "messages": [{"role": "assistant", "content": adv.OPENING}],
            "intent": "",
            "founder_profile": {},
            "cta_ready": False,
            "cta_offered": 0,
            "grounding_flags": [],
        }
    return StartResponse(
        session_id=sid,
        disclosure=DISCLOSURE,
        message=adv.OPENING,
    )


@app.post("/conversation/message", response_model=MessageResponse)
def message(req: MessageRequest) -> MessageResponse:
    with _LOCK:
        _check_global_cap()
        state = _get_session(req.session_id)
        user_turns = sum(1 for m in state["messages"] if m["role"] == "user")
        if user_turns >= settings.max_messages_per_session:
            raise HTTPException(status_code=429, detail="Session message limit reached.")
        state["messages"].append({"role": "user", "content": req.message})
        state["last_user_input"] = req.message

    result = GRAPH.invoke(state)

    with _LOCK:
        _SESSIONS[req.session_id] = result

    return MessageResponse(
        message=result.get("assistant_reply", ""),
        intent=result.get("intent", ""),
        recommendation_ready=bool(result.get("cta_ready", False)),
        grounding_flags=result.get("grounding_flags", []),
    )


@app.post("/capture", response_model=CaptureResponse)
def capture(req: CaptureRequest) -> CaptureResponse:
    _get_session(req.session_id)
    record = capture_mod.capture_lead(req.session_id, req.name, req.email, None)
    return CaptureResponse(
        captured=True,
        tool_source=record["tool_source"],
        recommendation_attached=False,
    )


# ── Frontend adapter endpoint ─────────────────────────────────────────────────
# The ss-advisor frontend sends:
#   { type, state_token, message, quick_action_id, turn_client_id, client }
# and expects back:
#   { reply, state_token, cta, suggestions, state }
#
# This endpoint translates between those two contracts so the frontend works
# without any changes. Session state is keyed by state_token (which the frontend
# stores in sessionStorage and echoes back each turn).

_ADAPTER_SESSIONS: dict[str, dict] = {}
_ADAPTER_LOCK = threading.Lock()


@app.post("/webhook/advisor-1")
async def advisor_webhook(request: Request):
    """Adapter for the ss-advisor frontend contract."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Analytics events — acknowledge and ignore
    if body.get("type") == "event":
        return {"ok": True}

    message_text = body.get("message", "").strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="message is required")

    state_token = body.get("state_token") or ""

    with _ADAPTER_LOCK:
        _check_global_cap()

        # Restore or create session
        if state_token and state_token in _ADAPTER_SESSIONS:
            state = _ADAPTER_SESSIONS[state_token]
        else:
            # New session — generate a token and initialise state
            state_token = uuid.uuid4().hex
            state = {
                "session_id": state_token,
                "messages": [{"role": "assistant", "content": adv.OPENING}],
                "intent": "",
                "founder_profile": {},
                "cta_ready": False,
                "cta_offered": 0,
                "grounding_flags": [],
            }
            _ADAPTER_SESSIONS[state_token] = state

        user_turns = sum(1 for m in state["messages"] if m["role"] == "user")
        if user_turns >= settings.max_messages_per_session:
            return {
                "reply": "We've covered a lot of ground. Book a free 30-minute strategy call to keep going with a real person: " + BOOK_URL,
                "state_token": state_token,
                "cta": [{"id": "consult", "label": "Book a free strategy call", "url": BOOK_URL, "kind": "link"}],
                "suggestions": [],
                "state": {"phase": "DONE", "handoff": True, "ask_contact": None, "closing": True},
            }

        state["messages"].append({"role": "user", "content": message_text})
        state["last_user_input"] = message_text

    # Run the graph outside the lock
    result = GRAPH.invoke(state)

    with _ADAPTER_LOCK:
        _ADAPTER_SESSIONS[state_token] = result

    reply = result.get("assistant_reply", "")
    cta_ready = bool(result.get("cta_ready", False))

    # Build CTA list for the frontend
    cta = []
    if cta_ready and BOOK_URL in reply:
        # Strip the raw URL from the reply text and surface it as a proper CTA button
        reply = reply.replace(BOOK_URL, "").strip().rstrip(":")
        cta = [{"id": "consult", "label": "Book a free strategy call", "url": BOOK_URL, "kind": "link"}]
    elif cta_ready:
        cta = [{"id": "consult", "label": "Book a free strategy call", "url": BOOK_URL, "kind": "link"}]

    return {
        "reply": reply,
        "state_token": state_token,
        "cta": cta,
        "suggestions": [],
        "state": {
            "phase": "CTA" if cta_ready else "DIAGNOSING",
            "handoff": False,
            "ask_contact": None,
            "closing": False,
        },
    }


# ── Notion webhook ────────────────────────────────────────────────────────────
import hashlib, hmac, threading as _threading

_SYNC_LOCK = _threading.Lock()
_SYNCING = False


def _run_sync():
    global _SYNCING
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import notion_page_sync
        notion_page_sync.main()
    except Exception as e:
        import logging
        logging.getLogger("webhook").error("sync failed: %s", e)
    finally:
        global _SYNCING
        _SYNCING = False


@app.post("/notion-webhook")
async def notion_webhook(request: Request):
    """Notion calls this when the KB page is edited. Triggers a re-sync."""
    global _SYNCING

    body = await request.json()
    if "challenge" in body:
        return {"challenge": body["challenge"]}

    with _SYNC_LOCK:
        if _SYNCING:
            return {"status": "sync already in progress"}
        _SYNCING = True

    t = _threading.Thread(target=_run_sync, daemon=True)
    t.start()
    return {"status": "sync started"}


@app.get("/sync-status")
def sync_status():
    return {"syncing": _SYNCING}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)