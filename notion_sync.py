from __future__ import annotations
import os
from typing import List, Tuple
from notion_client import Client

NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
NOTION_DB_ID = os.environ.get("NOTION_DB_ID", "")
COL_LAYER = "Layer"
COL_STAGE = "Stage"
COL_TYPE = "Type"
COL_CONTENT = "Content"

def _client():
    if not NOTION_TOKEN:
        raise RuntimeError("NOTION_TOKEN not set")
    return Client(auth=NOTION_TOKEN)

def _plain(prop):
    if not prop:
        return ""
    t = prop.get("type")
    if t == "title":
        return "".join(x.get("plain_text","") for x in prop.get("title",[])).strip()
    if t == "rich_text":
        return "".join(x.get("plain_text","") for x in prop.get("rich_text",[])).strip()
    if t == "select":
        sel = prop.get("select")
        return sel.get("name","").strip() if sel else ""
    return ""

def _data_source_id(nc, db_id):
    db = nc.databases.retrieve(db_id)
    sources = db.get("data_sources", [])
    return sources[0]["id"] if sources else db_id

def _query_all(nc, source_id, is_ds):
    rows = []
    cursor = None
    path = f"data_sources/{source_id}/query" if is_ds else f"databases/{source_id}/query"
    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        res = nc.request(path=path, method="POST", body=body)
        rows.extend(res.get("results", []))
        if not res.get("has_more"):
            break
        cursor = res.get("next_cursor")
    return rows

def fetch_chunks():
    nc = _client()
    if not NOTION_DB_ID:
        raise RuntimeError("NOTION_DB_ID not set")
    source_id = _data_source_id(nc, NOTION_DB_ID)
    is_ds = source_id != NOTION_DB_ID
    rows = _query_all(nc, source_id, is_ds)
    chunks = []
    for r in rows:
        props = r.get("properties", {})
        name = ""
        for v in props.values():
            if v.get("type") == "title":
                name = _plain(v); break
        layer = _plain(props.get(COL_LAYER, {}))
        stage = _plain(props.get(COL_STAGE, {})) or "any"
        typ = _plain(props.get(COL_TYPE, {})) or "guidance"
        content = _plain(props.get(COL_CONTENT, {}))
        if not content:
            continue
        chunks.append(({"layer": layer, "stage": stage, "type": typ, "name": name}, content))
    return chunks
