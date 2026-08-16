"""Lead capture. In the draft build this records to memory and logs; production wires
this to the shared trio capture infrastructure with tool_source tagging and the
snapshot attached as the durable record.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from config import settings
from app.schemas import Snapshot

logger = logging.getLogger("capture")

_CAPTURES: list[dict] = []


def capture_lead(session_id: str, name: str, email: str, snapshot: Snapshot | None) -> dict:
    record = {
        "session_id": session_id,
        "name": name,
        "email": email,
        "tool_source": settings.tool_source,
        "snapshot_attached": snapshot is not None,
        "snapshot": snapshot.model_dump() if snapshot else None,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    _CAPTURES.append(record)
    logger.info("captured lead %s", json.dumps({k: record[k] for k in ("session_id", "email", "tool_source", "snapshot_attached")}))
    return record


def all_captures() -> list[dict]:
    return list(_CAPTURES)
