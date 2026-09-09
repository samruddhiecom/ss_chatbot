"""
notion_page_sync.py — reads a Notion page, chunks by H2/H3 headings, ingests into ChromaDB.

PMs edit the Notion page like a normal document.
Run this script to update the bot's knowledge base.

Usage:
    python notion_page_sync.py
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOKEN   = os.environ["NOTION_TOKEN"]
PAGE_ID = os.environ.get("NOTION_PAGE_ID", "3d48ba072e19805a96d3f42b67c9a6e0")

from notion_client import Client
from app.kb import store

nc = Client(auth=TOKEN)

# ── 1. Fetch all blocks recursively ─────────────────────────────────────────
def get_blocks(block_id):
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

def block_text(block):
    bt = block.get("type", "")
    node = block.get(bt, {})
    if isinstance(node, dict):
        rich = node.get("rich_text", [])
        return "".join(r.get("plain_text", "") for r in rich).strip()
    return ""

# ── 2. Chunk by heading ───────────────────────────────────────────────────────
HEADING_TYPES = {"heading_1", "heading_2", "heading_3"}

def chunk_blocks(blocks):
    """Split blocks into chunks at every heading boundary."""
    chunks = []
    current_heading = "Introduction"
    current_level  = "heading_1"
    current_lines  = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append({
                "heading": current_heading,
                "level":   current_level,
                "text":    text,
            })

    for b in blocks:
        bt = block_text(b)
        btype = b.get("type", "")
        if btype in HEADING_TYPES:
            flush()
            current_heading = bt
            current_level   = btype
            current_lines   = []
        elif bt:
            current_lines.append(bt)

    flush()
    return chunks

# ── 3. Tag chunks with category from the KB doc's own metadata ───────────────
GUARDRAIL_KEYWORDS = {"guardrail", "never do", "never", "must not", "do not"}
VOICE_KEYWORDS     = {"persona", "style rule", "voice", "tone", "advisor", "opening line"}
ROUTING_KEYWORDS   = {"routing", "escalate", "handoff", "intent"}
DEFERRAL_KEYWORDS  = {"legal", "tax", "accounting", "investment", "visa", "medical"}

def classify_chunk(heading: str, text: str):
    h = heading.lower()
    t = text.lower()
    combined = h + " " + t
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
    if "service" in h or "digital marketing" in h or "bookkeeping" in h or "automation" in h:
        return "service"
    if "pricing" in h or "commercial" in h or "bundle" in h:
        return "pricing"
    if "process" in h or "phase" in h:
        return "process"
    if "proof" in h or "testimonial" in h or "credibility" in h:
        return "social_proof"
    if "objection" in h:
        return "objection"
    if "who we" in h or "good fit" in h or "not a fit" in h:
        return "fit"
    if "positioning" in h or "messaging" in h or "why us" in h or "trust" in h:
        return "positioning"
    if "snapshot" in h or "company" in h or "elevator" in h:
        return "company_fact"
    if "qualification" in h or "lead" in h:
        return "lead_qualification"
    if "example" in h or "worked" in h:
        return "example"
    if "glossary" in h:
        return "glossary"
    if "site map" in h or "links" in h:
        return "navigation"
    return "guidance"

# ── 4. Ingest ─────────────────────────────────────────────────────────────────
print(f"Fetching page {PAGE_ID}...")
blocks = get_blocks(PAGE_ID)
print(f"  Got {len(blocks)} top-level blocks")

chunks = chunk_blocks(blocks)
print(f"  Split into {len(chunks)} chunks")

store.reset_collection()
ids, docs, metas = [], [], []

for i, chunk in enumerate(chunks):
    chunk_type = classify_chunk(chunk["heading"], chunk["text"])
    # Voice/persona chunks stay out of retrieval corpus (same as Layer C rule)
    if chunk_type == "voice":
        print(f"  [SKIP - voice] {chunk['heading']}")
        continue
    enriched = f"[{chunk_type} | {chunk['heading']}] {chunk['text']}"
    ids.append(f"notion-chunk-{i}")
    docs.append(enriched)
    metas.append({
        "heading": chunk["heading"],
        "level":   chunk["level"],
        "type":    chunk_type,
        "source":  "notion-page",
    })
    print(f"  [{chunk_type}] {chunk['heading'][:60]}")

store.add(ids, docs, metas)
print(f"\nDone. Ingested {len(docs)} chunks into ChromaDB.")
print(f"Collection now holds: {store.count()} chunks")