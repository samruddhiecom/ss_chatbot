"""Ingest the draft knowledge base into ChromaDB.

Run from the project root:  python scripts/ingest_kb.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.kb.ingest import ingest  # noqa: E402

if __name__ == "__main__":
    total = ingest(reset=True)
    print(f"Ingested draft KB (Layers A + B). Collection now holds {total} chunks.")
    print("Layer C (brand voice) is intentionally excluded from the corpus.")
