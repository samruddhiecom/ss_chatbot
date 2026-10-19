"""Conversation persistence to the Supabase conversations table.

Upserts on each turn so transcript, turn count, stage and profile stay fresh.
Generates a short LLM summary of the conversation after each turn.

Env vars:
  SUPABASE_URL       -- Supabase project URL
  SUPABASE_ANON_KEY  -- Supabase anon key
  GROQ_API_KEY       -- Used for summary generation
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger("ss_advisor")

_SUMMARY_SYSTEM = (
    "You summarise a founder's conversation with the Simplified Startup AI Advisor in 2-3 sentences. "
    "Cover: what business they're building, their stage, their main bottleneck, what they've tried, "
    "and whether they booked or expressed interest in a strategy call. "
    "Be specific and factual — use their own words where possible. No fluff."
)


def _generate_summary(transcript: str, profile: dict) -> str:
    """Generate a 2-3 sentence summary using Groq. Returns empty string on failure."""
    try:
        from app import llm
        if not llm.settings.has_llm:
            return ""
        profile_text = json.dumps(profile or {}, ensure_ascii=False)
        return llm.generate(
            system=_SUMMARY_SYSTEM,
            user=f"Founder profile: {profile_text}\n\nConversation:\n{transcript}",
            temperature=0.2,
        )
    except Exception as e:
        logger.error("Summary generation failed: %s", e)
        return ""


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
    """Upsert the conversation row in Supabase with a generated summary.

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

        # Only generate summary when there are at least 2 user turns (enough context).
        summary = ""
        if turn_count >= 2:
            summary = _generate_summary(transcript, profile_json)

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
            "summary": summary,
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