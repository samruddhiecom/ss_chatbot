"""Lead capture -- saves name, email, mobile to the SS Advisor Leads Notion database."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from notion_client import Client

LEADS_DB_ID = os.environ.get("NOTION_LEADS_DB_ID", "3f18ba072e1980509046dce419cd38fe")


def save_lead(name: str, email: str, mobile: str, session_id: str) -> bool:
    """Save a lead to the Notion Leads database. Returns True on success."""
    try:
        token = os.environ["NOTION_TOKEN"]
        nc = Client(auth=token)
        nc.pages.create(
            parent={"database_id": LEADS_DB_ID},
            properties={
                "Name": {"title": [{"text": {"content": name.strip()}}]},
                "Email": {"email": email.strip()},
                "Mobile": {"phone_number": mobile.strip()},
                "Submitted at": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
                "Session ID": {"rich_text": [{"text": {"content": session_id}}]},
            },
        )
        return True
    except Exception as e:
        import logging
        logging.getLogger("ss_advisor").error("Lead capture failed: %s", e)
        return False