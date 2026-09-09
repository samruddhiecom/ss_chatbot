"""Lead capture. Records to memory and logs in draft build.
Production wires this to the shared trio capture infrastructure.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from config import settings

logger = logging.getLogger("capture")
_CAPTURES: list[dict] = []


def capture_lead(session_id: str, name: str, email: str, recommendation=None) -> dict:
    record = {
        "session_id": session_id,
        "name": name,
        "email": email,
        "tool_source": settings.tool_source,
        "snapshot_attached": recommendation is not None,
        "recommendation": recommendation.model_dump() if recommendation and hasattr(recommendation, "model_dump") else recommendation,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    _CAPTURES.append(record)
    logger.info("captured lead %s", json.dumps({
        k: record[k] for k in ("session_id", "email", "tool_source", "snapshot_attached")
    }))
    return record
