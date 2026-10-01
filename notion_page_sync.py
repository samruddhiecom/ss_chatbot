"""
notion_page_sync.py — syncs the SS Advisor Knowledge Base (Notion DATABASE) into ChromaDB.

Changes vs original:
  1. Reads from the Notion DATABASE (NOTION_DB_ID env var), not a single page.
  2. Enforces Visibility=Public AND Status=Approved filter.
  3. Reads row body blocks via blocks.children.list.
  4. Heading-based chunking with sentence-level overlap.
  5. Pricing rows ARE now indexed (Option B) — bot can surface KB-approved prices
     when directly asked. Chunks are tagged type='pricing' so they route correctly.

Usage:
    python notion_page_sync.py            # manual run
    Called automatically via /notion-webhook endpoint in main.py

Environment variables required:
    NOTION_TOKEN   -- Notion integration token
    NOTION_DB_ID   -- Database ID (e74924e630fa46e89791c6eb44604c42)
"""
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOKEN = os.environ["NOTION_TOKEN"]
DB_ID = os.environ.get("NOTION_DB_ID", "e74924e630fa46e89791c6eb44604c42")

from notion_client import Client
from app.kb import store

nc = Client(auth=TOKEN)


# ── 1. Query the database — approved public rows only ────────────────────────

def _plain(prop: dict) -> str:
    if not prop:
        return ""
    t = prop.get("type", "")
    if t == "title":
        return "".join(r.get("plain_text", "") for r in prop.get("title", [])).strip()
    if t == "rich_text":
        return "".join(r.get("plain_text", "") for r in prop.get("rich_text", [])).strip()
    if t == "select":
        sel = prop.get("select")
        return sel.get("name", "").strip() if sel else ""
    if t == "checkbox":
        return str(prop.get("checkbox", False))
    if t == "status":
        s = prop.get("status")
        return s.get("name", "").strip() if s else ""
    return ""


def fetch_approved_rows() -> list[dict]:
    """Return all DB rows where Visibility=Public AND Status=Approved."""
    rows = []
    cursor = None
    while True:
        kwargs = {
            "database_id": DB_ID,
            "page_size": 100,
            "filter": {
                "and": [
                    {"property": "Visibility", "select": {"equals": "Public"}},
                    {"property": "Status", "status": {"equals": "Approved"}},
                ]
            },
        }
        if cursor:
            kwargs["start_cursor"] = cursor
        res = nc.databases.query(**kwargs)
        rows.extend(res.get("results", []))
        if not res.get("has_more"):
            break
        cursor = res.get("next_cursor")
    return rows


# ── 2. Fetch page body blocks for a row ──────────────────────────────────────

def _get_blocks(block_id: str) -> list[dict]:
    results = []
    cursor = None
    while True:
        kwargs = {"block_id": block_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        res = nc.blocks.children.list(**kwargs)
        results.extend(res.get("results", []))
        if not res.get("has_more"):
            break
        cursor = res.get("next_cursor")
    return results


def _block_text(block: dict) -> str:
    bt = block.get("type", "")
    node = block.get(bt, {})
    if isinstance(node, dict):
        rich = node.get("rich_text", [])
        return "".join(r.get("plain_text", "") for r in rich).strip()
    return ""


# ── 3. Chunking with sentence-level overlap ───────────────────────────────────

HEADING_TYPES = {"heading_1", "heading_2", "heading_3"}
SENT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def chunk_blocks(blocks: list[dict], title: str) -> list[dict]:
    chunks = []
    current_heading = title
    current_level = "heading_1"
    current_lines: list[str] = []
    carry: str = ""

    def flush(next_heading: str = "") -> None:
        nonlocal carry
        text = "\n".join(current_lines).strip()
        if text:
            full_text = (carry + " " + text).strip() if carry else text
            chunks.append({
                "heading": current_heading,
                "level": current_level,
                "text": full_text,
            })
            sentences = _split_sentences(text)
            carry = sentences[-1] if sentences else ""

    for b in blocks:
        btype = b.get("type", "")
        bt = _block_text(b)
        if btype in HEADING_TYPES:
            flush()
            current_heading = bt or current_heading
            current_level = btype
            current_lines = []
        elif bt:
            current_lines.append(bt)

    flush()
    return chunks


# ── 4. Category classification ────────────────────────────────────────────────

GUARDRAIL_KEYWORDS = {"guardrail", "never do", "never", "must not", "do not"}
VOICE_KEYWORDS = {"persona", "style rule", "voice", "tone", "advisor", "opening line"}
ROUTING_KEYWORDS = {"routing", "escalate", "handoff", "intent", "cta"}
DEFERRAL_KEYWORDS = {"legal", "tax", "accounting", "investment", "visa", "medical"}


def classify_chunk(heading: str, text: str, contains_pricing: bool = False) -> str:
    if contains_pricing:
        return "pricing"
    h = heading.lower()
    if any(k in h for k in GUARDRAIL_KEYWORDS):
        return "boundary"
    if any(k in h for k in VOICE_KEYWORDS):
        return "voice"
    if any(k in h for k in ROUTING_KEYWORDS):
        return "routing"
    if any(k in h for k in DEFERRAL_KEYWORDS):
        return "deferral_language"
    if "faq" in h or "frequently" in h:
        return "faq"
    if any(k in h for k in ("service", "digital marketing", "bookkeeping", "automation",
                              "talent", "branding", "website", "sales", "advisory")):
        return "service"
    if any(k in h for k in ("pricing", "commercial", "bundle", "discount", "standalone")):
        return "pricing"
    if any(k in h for k in ("process", "phase", "growth plan")):
        return "process"
    if any(k in h for k in ("proof", "testimonial", "credibility", "published")):
        return "social_proof"
    if "objection" in h:
        return "objection"
    if any(k in h for k in ("who this", "good fit", "not a fit", "fit")):
        return "fit"
    if any(k in h for k in ("positioning", "messaging", "why us", "trust", "compare")):
        return "positioning"
    if any(k in h for k in ("snapshot", "company", "elevator")):
        return "company_fact"
    if any(k in h for k in ("qualification", "lead")):
        return "lead_qualification"
    if any(k in h for k in ("example", "worked")):
        return "example"
    if "glossary" in h:
        return "glossary"
    if any(k in h for k in ("site map", "links")):
        return "navigation"
    return "guidance"


# ── 5. Ingest ─────────────────────────────────────────────────────────────────

def main() -> None:
    rows = fetch_approved_rows()
    print(f"Fetched {len(rows)} approved public rows from DB {DB_ID}")

    store.reset_collection()
    ids, docs, metas = [], [], []
    chunk_index = 0

    for row in rows:
        props = row.get("properties", {})
        row_id = row["id"]

        title = ""
        for v in props.values():
            if v.get("type") == "title":
                title = _plain(v)
                break

        # Option B: pricing rows are now indexed, tagged as type='pricing'
        contains_pricing = _plain(props.get("Contains pricing", {})) == "True"

        blocks = _get_blocks(row_id)
        if not blocks:
            print(f"  [SKIP - empty] {title}")
            continue

        chunks = chunk_blocks(blocks, title)
        print(f"  [{len(chunks)} chunks] {title}")

        for chunk in chunks:
            chunk_type = classify_chunk(
                chunk["heading"], chunk["text"], contains_pricing=contains_pricing
            )

            # Voice/persona chunks stay out of retrieval corpus
            if chunk_type == "voice":
                continue

            enriched = f"[{chunk_type} | {chunk['heading']}] {chunk['text']}"
            ids.append(f"notion-row-{row_id[:8]}-chunk-{chunk_index}")
            docs.append(enriched)
            metas.append({
                "heading": chunk["heading"],
                "level": chunk["level"],
                "type": chunk_type,
                "source": "notion-db",
                "row_title": title,
            })
            chunk_index += 1

    store.add(ids, docs, metas)
    print(f"\nDone. Ingested {len(docs)} chunks from {len(rows)} rows into ChromaDB.")
    print(f"Collection now holds: {store.count()} chunks")


if __name__ == "__main__":
    main()