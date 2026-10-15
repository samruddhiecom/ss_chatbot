"""Lead capture -- saves name, email, mobile to Supabase leads table."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("ss_advisor")


def save_lead(name: str, email: str, mobile: str, session_id: str) -> bool:
    """Save a lead to Supabase. Returns True on success."""
    try:
        from app.supabase_client import get_client
        sb = get_client()
        if sb is None:
            return False
        sb.table("leads").insert({
            "session_id": session_id or "",
            "name": name.strip(),
            "email": email.strip(),
            "mobile": mobile.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as e:
        logger.error("Lead capture failed: %s", e)
        return False