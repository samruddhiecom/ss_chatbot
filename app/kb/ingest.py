"""Ingest the draft knowledge base into ChromaDB.

Chunking is deterministic: each `### [k=v; k=v]` header starts a new chunk and
carries its own metadata. Each chunk is context-enriched with a one-line header
before embedding (Anthropic Contextual Retrieval pattern), which measurably helps
retrieval on small corpora.

Layer C (brand voice) is intentionally NOT ingested; it lives in the prompt.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from config import DRAFT_KB_DIR
from app.kb import store

HEADER_RE = re.compile(r"^###\s*\[(.+?)\]\s*$")

# Files that go into the retrieval corpus. layer_c_voice.md is excluded by design.
CORPUS_FILES = ["layer_a_guidance.md", "layer_b_company.md"]


def _parse_meta(raw: str) -> dict:
    meta = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def parse_file(path: Path) -> List[Tuple[dict, str]]:
    chunks: List[Tuple[dict, str]] = []
    meta: dict | None = None
    buf: List[str] = []

    def flush():
        if meta is not None:
            text = "\n".join(buf).strip()
            if text:
                chunks.append((meta, text))

    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADER_RE.match(line.strip())
        if m:
            flush()
            meta = _parse_meta(m.group(1))
            buf = []
        elif line.strip().startswith("#") and meta is None:
            # top-of-file comments before the first chunk header
            continue
        else:
            buf.append(line)
    flush()
    return chunks


def _context_header(meta: dict) -> str:
    layer = meta.get("layer", "?")
    stage = meta.get("stage", "any")
    typ = meta.get("type", "guidance")
    return f"[Layer {layer} | stage: {stage} | {typ}]"


def ingest(reset: bool = True) -> int:
    if reset:
        store.reset_collection()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []

    for fname in CORPUS_FILES:
        path = DRAFT_KB_DIR / fname
        for i, (meta, text) in enumerate(parse_file(path)):
            enriched = f"{_context_header(meta)} {text}"
            chunk_id = f"{meta.get('layer','X')}-{meta.get('stage','any')}-{meta.get('type','g')}-{i}"
            ids.append(chunk_id)
            documents.append(enriched)
            metadatas.append({
                "layer": meta.get("layer", ""),
                "stage": meta.get("stage", "any"),
                "type": meta.get("type", "guidance"),
                "source": fname,
            })

    store.add(ids, documents, metadatas)
    return store.count()


if __name__ == "__main__":
    total = ingest(reset=True)
    print(f"Ingested draft KB. Collection now holds {total} chunks.")
