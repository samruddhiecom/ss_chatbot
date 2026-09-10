"""FastAPI engine for the SS AI Advisor.

Endpoints:
  POST /conversation/start    — disclosure + opening message
  POST /conversation/message  — send a message, get a reply with CTA when ready
  POST /capture               — submit lead (name + email)
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
        notion_page_sync.NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
        notion_page_sync.PAGE_ID = os.environ.get("NOTION_PAGE_ID", "3d48ba072e19805a96d3f42b67c9a6e0")
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
    from fastapi import Request
    global _SYNCING

    # Notion sends a verification challenge on first setup
    body = await request.json()
    if "challenge" in body:
        return {"challenge": body["challenge"]}

    # Deduplicate — only one sync at a time
    with _SYNC_LOCK:
        if _SYNCING:
            return {"status": "sync already in progress"}
        _SYNCING = True

    # Run sync in background so webhook returns immediately
    t = _threading.Thread(target=_run_sync, daemon=True)
    t.start()
    return {"status": "sync started"}


@app.get("/sync-status")
def sync_status():
    return {"syncing": _SYNCING}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)