"""FastAPI engine for the SS AI Advisor.

Endpoints:
  POST /conversation/start    -- disclosure + opening message
  POST /conversation/message  -- send a message, get a reply (+ chips) with CTA when ready
  POST /capture               -- submit lead (name + email)
  POST /lead                  -- lead gate submission (name + email + mobile)
  POST /webhook/advisor-1     -- adapter for the ss-advisor frontend contract
  GET  /health
  GET  /sync-status
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
import uuid
from datetime import date, datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from app import capture as capture_mod
from app import conversation_store
from app.graph import GRAPH
from app.schemas import (
    CaptureRequest,
    CaptureResponse,
    MessageRequest,
    MessageResponse,
    StartResponse,
)
from app import advisor as adv

logger = logging.getLogger("ss_advisor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="SS AI Advisor Engine", version="2.3.0")

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


def _persist_turn(sid: str, state: dict) -> None:
    """Fire-and-forget persistence to Notion Conversations DB. Never blocks."""
    try:
        conversation_store.persist_async(
            session_id=sid,
            messages=state.get("messages", []),
            stage=state.get("conv_stage", ""),
            cta_offered=int(state.get("cta_offered", 0) or 0),
            profile=state.get("founder_profile", {}),
            lead_name=state.get("lead_name", ""),
            lead_email=state.get("lead_email", ""),
            started_at=state.get("started_at", ""),
        )
    except Exception as e:
        logger.error("Persist dispatch failed: %s", e)


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
            "chips": [],
            "grounding_flags": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "lead_name": "",
            "lead_email": "",
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

    try:
        result = GRAPH.invoke(state)
    except Exception as e:
        logger.error("Graph invoke failed for %s: %s\n%s", req.session_id, e, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Advisor engine error. Please retry.")

    with _LOCK:
        # Carry across the lead and started_at fields that the graph does not touch.
        result["started_at"] = state.get("started_at", "")
        result["lead_name"] = state.get("lead_name", "")
        result["lead_email"] = state.get("lead_email", "")
        _SESSIONS[req.session_id] = result

    _persist_turn(req.session_id, result)

    phase = result.get("conv_stage", "")
    # Explore button rides along with cta, soft_close, close so the founder always has a
    # low-friction way to look at the website instead of being funneled only to booking.
    show_explore = phase in ("cta", "soft_close", "close")
    show_success = phase == "success"

    # In the success phase the message shouldn't carry URLs — buttons do.
    reply_text = result.get("assistant_reply", "")
    if show_success:
        import re as _re
        reply_text = _re.sub(r"https?://\S+", "", reply_text).strip().rstrip(":").strip()

    return MessageResponse(
        message=reply_text,
        intent=result.get("intent", ""),
        recommendation_ready=bool(result.get("cta_ready", False)),
        chips=result.get("chips", []) or [],
        grounding_flags=result.get("grounding_flags", []),
        phase=phase,
        show_explore_button=show_explore,
        show_success_stories_button=show_success,
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


# ── Lead capture endpoint ────────────────────────────────────────────────────
from pydantic import BaseModel as _BaseModel

class LeadRequest(_BaseModel):
    name: str
    email: str
    mobile: str
    session_id: str = ""

@app.post("/lead")
def lead(req: LeadRequest):
    from app.lead_capture import save_lead
    ok = save_lead(req.name, req.email, req.mobile, req.session_id)
    # Stash the lead on the session so it lands in the Conversations DB too.
    if req.session_id:
        with _LOCK:
            s = _SESSIONS.get(req.session_id)
            if s is not None:
                s["lead_name"] = req.name
                s["lead_email"] = req.email
    return {"saved": ok}


# ── Frontend adapter endpoint ─────────────────────────────────────────────────
_ADAPTER_SESSIONS: dict[str, dict] = {}
_ADAPTER_LOCK = threading.Lock()


@app.post("/webhook/advisor-1")
async def advisor_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if body.get("type") == "event":
        return {"ok": True}

    message_text = body.get("message", "").strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="message is required")

    state_token = body.get("state_token") or ""

    with _ADAPTER_LOCK:
        _check_global_cap()
        if state_token and state_token in _ADAPTER_SESSIONS:
            state = _ADAPTER_SESSIONS[state_token]
        else:
            state_token = uuid.uuid4().hex
            state = {
                "session_id": state_token,
                "messages": [{"role": "assistant", "content": adv.OPENING}],
                "intent": "",
                "founder_profile": {},
                "cta_ready": False,
                "cta_offered": 0,
                "chips": [],
                "grounding_flags": [],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "lead_name": "",
                "lead_email": "",
            }
            _ADAPTER_SESSIONS[state_token] = state

        user_turns = sum(1 for m in state["messages"] if m["role"] == "user")
        if user_turns >= settings.max_messages_per_session:
            return {
                "reply": "We've covered a lot of ground. Book a free 30-minute strategy call to keep going with a real person.",
                "state_token": state_token,
                "cta": [{"id": "consult", "label": "Book a free strategy call", "url": BOOK_URL, "kind": "link"}],
                "suggestions": [],
                "state": {"phase": "DONE", "handoff": True, "ask_contact": None, "closing": True},
            }

        state["messages"].append({"role": "user", "content": message_text})
        state["last_user_input"] = message_text

    try:
        result = GRAPH.invoke(state)
    except Exception as e:
        logger.error("Adapter graph invoke failed: %s\n%s", e, traceback.format_exc())
        return {
            "reply": "I hit a snag just now. Could you say that again, or rephrase it?",
            "state_token": state_token,
            "cta": [],
            "suggestions": [],
            "state": {"phase": "ERROR", "handoff": False, "ask_contact": None, "closing": False},
        }

    with _ADAPTER_LOCK:
        result["started_at"] = state.get("started_at", "")
        result["lead_name"] = state.get("lead_name", "")
        result["lead_email"] = state.get("lead_email", "")
        _ADAPTER_SESSIONS[state_token] = result

    _persist_turn(state_token, result)

    reply = result.get("assistant_reply", "")
    cta_ready = bool(result.get("cta_ready", False))
    chips = result.get("chips", []) or []
    phase = result.get("conv_stage", "DIAGNOSING")

    # Strip ANY URL from close/cta replies — the CTA card carries the correct link.
    import re as _re
    _URL_RE = _re.compile(r"https?://\S+")
    if cta_ready or phase in ("success",):
        reply = _URL_RE.sub("", reply).strip().rstrip(":").strip()

    cta = []
    if cta_ready:
        cta = [{"id": "consult", "label": "Book a free strategy call", "url": BOOK_URL, "kind": "link"}]

    return {
        "reply": reply,
        "state_token": state_token,
        "cta": cta,
        "suggestions": chips,
        "state": {
            "phase": phase,
            "handoff": False,
            "ask_contact": None,
            "closing": cta_ready,
        },
    }


# ── Notion sync ───────────────────────────────────────────────────────────────
# Memory posture on Render free tier (512 MB):
#   - First sync delayed 10 minutes so startup finishes cleanly and the first
#     user requests do not overlap with the sync's embedder spike.
#   - Sync runs in its own thread, then explicitly releases the fastembed model.
#   - gc.collect() after sync to return baseline memory before the next request.
_SYNC_LOCK = threading.Lock()
_SYNCING = False
SYNC_INTERVAL_SECONDS = 60 * 60  # 60 minutes (was 30)
FIRST_SYNC_DELAY_SECONDS = 10 * 60  # 10 minutes (was 60s)


def _run_sync(label: str = "scheduled") -> None:
    global _SYNCING
    import gc as _gc
    logger.info("Notion KB sync started (%s)", label)
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import notion_page_sync
        notion_page_sync.main()
        logger.info("Notion KB sync completed (%s)", label)
    except Exception as e:
        logger.error("Notion KB sync failed (%s): %s", label, e)
    finally:
        # Release the embedder so RSS drops back to baseline between syncs.
        try:
            from app.kb import store as _store
            _store.drop_embedder()
        except Exception:
            pass
        _gc.collect()
        _SYNCING = False


def _schedule_sync() -> None:
    """Re-schedule the next sync after SYNC_INTERVAL_SECONDS."""
    t = threading.Timer(SYNC_INTERVAL_SECONDS, _scheduled_sync)
    t.daemon = True
    t.start()


def _scheduled_sync() -> None:
    global _SYNCING
    with _SYNC_LOCK:
        if _SYNCING:
            logger.info("Scheduled sync skipped — sync already in progress")
            _schedule_sync()
            return
        _SYNCING = True
    threading.Thread(target=lambda: [_run_sync("scheduled"), _schedule_sync()], daemon=True).start()


@app.on_event("startup")
async def startup_event():
    """Kick off the first scheduled sync and set the hourly polling cadence.
    The first sync is delayed well past startup so user requests get a stable
    process rather than one still spiking from embedder load + ingest."""
    logger.info("App started. First Notion KB sync in %d seconds.", FIRST_SYNC_DELAY_SECONDS)
    t = threading.Timer(FIRST_SYNC_DELAY_SECONDS, _scheduled_sync)
    t.daemon = True
    t.start()


@app.post("/notion-webhook")
async def notion_webhook(request: Request):
    """Manual trigger — also used for testing. Fires the sync immediately."""
    global _SYNCING
    body = await request.json()
    if "challenge" in body:
        return {"challenge": body["challenge"]}

    with _SYNC_LOCK:
        if _SYNCING:
            return {"status": "sync already in progress"}
        _SYNCING = True

    threading.Thread(target=_run_sync, args=("webhook",), daemon=True).start()
    return {"status": "sync started"}


@app.get("/sync-status")
def sync_status():
    return {"syncing": _SYNCING}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)