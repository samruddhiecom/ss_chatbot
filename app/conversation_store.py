"""Conversation persistence to the Supabase conversations table.

Upserts on each turn so transcript, turn count, stage and profile stay fresh.

Env vars:
  SUPABASE_URL       -- Supabase project URL
  SUPABASE_ANON_KEY  -- Supabase anon key
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger("ss_advisor")


def _transcript_text(messages: list[dict]) -> str:
    lines = []
    for m in messages or []:
        who = "FOUNDER" if m.get("role") == "user" else "ADVISOR"
        lines.append(f"{who}: {m.get('content', '')}")
    return "\n".join(lines)


def upsert_conversation(
    session_id: str,
    messages: list[dict],
    stage: str = "",
    cta_offered: int = 0,
    profile: dict | None = None,
    lead_name: str = "",
    lead_email: str = "",
    started_at: str = "",
) -> bool:
    """Upsert the conversation row in Supabase.

    Returns True on success. Never raises — persistence failure must not
    break the chat endpoint.
    """
    try:
        from app.supabase_client import get_client
        sb = get_client()
        if sb is None:
            return False

        transcript = _transcript_text(messages)
        turn_count = sum(1 for m in (messages or []) if m.get("role") == "user")
        profile_json = profile or {}
        booked_cta = bool(cta_offered and stage in ("cta", "close"))
        now = datetime.now(timezone.utc).isoformat()

        row = {
            "session_id": session_id,
            "lead_name": lead_name or "",
            "lead_email": lead_email or "",
            "transcript": transcript,
            "turn_count": turn_count,
            "stage_reached": stage if stage in ("discover", "cta", "objection", "close", "soft_close", "post_cta", "success") else "",
            "booked_cta": booked_cta,
            "final_profile": json.dumps(profile_json, ensure_ascii=False),
            "last_updated": now,
        }
        if started_at:
            row["started_at"] = started_at

        sb.table("conversations").upsert(row, on_conflict="session_id").execute()
        return True
    except Exception as e:
        logger.error("Conversation persist failed for %s: %s", session_id, e)
        return False


def persist_async(**kwargs) -> None:
    """Fire and forget — never blocks the chat response."""
    threading.Thread(target=upsert_conversation, kwargs=kwargs, daemon=True).start()