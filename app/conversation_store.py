"""Conversation persistence to the SS Advisor Conversations Notion database.

Writes a conversation row per session, upserting on each turn so the full
transcript, turn count, stage reached, and final profile stay fresh.

Env vars:
  NOTION_TOKEN                   -- Notion integration token (reused from lead_capture)
  NOTION_CONVERSATIONS_DB_ID     -- Database ID of the SS Advisor Conversations DB
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone

from notion_client import Client

logger = logging.getLogger("ss_advisor")

CONVERSATIONS_DB_ID = os.environ.get(
    "NOTION_CONVERSATIONS_DB_ID",
    "3f78ba072e1980babd17d75ec98f9c3f",
)

# session_id -> notion page_id, kept in-memory so we can patch instead of re-creating.
_PAGE_INDEX: dict[str, str] = {}
_INDEX_LOCK = threading.Lock()

# Notion rich_text blocks cap at 2000 chars each; keep whole transcript under 50 blocks.
_MAX_BLOCK_CHARS = 1900
_MAX_BLOCKS = 50


def _client() -> Client | None:
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        return None
    return Client(auth=token)


def _chunk(text: str) -> list[dict]:
    if not text:
        return [{"text": {"content": ""}}]
    blocks = []
    s = text
    while s and len(blocks) < _MAX_BLOCKS:
        blocks.append({"text": {"content": s[:_MAX_BLOCK_CHARS]}})
        s = s[_MAX_BLOCK_CHARS:]
    if s and blocks:
        blocks[-1]["text"]["content"] = (
            blocks[-1]["text"]["content"][: _MAX_BLOCK_CHARS - 20] + " …[truncated]"
        )
    return blocks


def _transcript_text(messages: list[dict]) -> str:
    lines = []
    for m in messages or []:
        who = "FOUNDER" if m.get("role") == "user" else "ADVISOR"
        lines.append(f"{who}: {m.get('content', '')}")
    return "\n".join(lines)


def _properties(
    session_id: str,
    lead_name: str,
    lead_email: str,
    started_at: str,
    turn_count: int,
    stage: str,
    booked_cta: bool,
    transcript: str,
    profile_json: str,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    props = {
        "Session ID": {"title": [{"text": {"content": session_id}}]},
        "Last Updated": {"date": {"start": now}},
        "Turn Count": {"number": turn_count},
        "Booked CTA": {"checkbox": booked_cta},
        "Full Transcript": {"rich_text": _chunk(transcript)},
        "Final Profile": {"rich_text": _chunk(profile_json)},
    }
    if lead_name:
        props["Lead Name"] = {"rich_text": [{"text": {"content": lead_name}}]}
    if lead_email:
        props["Lead Email"] = {"email": lead_email}
    if started_at:
        props["Started At"] = {"date": {"start": started_at}}
    if stage in ("discover", "cta", "objection", "close"):
        props["Stage Reached"] = {"select": {"name": stage}}
    return props


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
    """Create or update the Notion row for this session.

    Returns True on success. Never raises -- a persistence failure must not
    break the chat endpoint.
    """
    try:
        nc = _client()
        if nc is None:
            return False

        transcript = _transcript_text(messages)
        turn_count = sum(1 for m in (messages or []) if m.get("role") == "user")
        profile_json = json.dumps(profile or {}, ensure_ascii=False, indent=2)
        booked_cta = bool(cta_offered and stage in ("cta", "close"))

        props = _properties(
            session_id=session_id,
            lead_name=lead_name,
            lead_email=lead_email,
            started_at=started_at,
            turn_count=turn_count,
            stage=stage,
            booked_cta=booked_cta,
            transcript=transcript,
            profile_json=profile_json,
        )

        with _INDEX_LOCK:
            page_id = _PAGE_INDEX.get(session_id)

        if page_id:
            nc.pages.update(page_id=page_id, properties=props)
            return True

        page = nc.pages.create(
            parent={"database_id": CONVERSATIONS_DB_ID},
            properties=props,
        )
        with _INDEX_LOCK:
            _PAGE_INDEX[session_id] = page["id"]
        return True
    except Exception as e:
        logger.error("Conversation persist failed for %s: %s", session_id, e)
        return False


def persist_async(**kwargs) -> None:
    """Fire and forget — never blocks the chat response."""
    threading.Thread(target=upsert_conversation, kwargs=kwargs, daemon=True).start()
