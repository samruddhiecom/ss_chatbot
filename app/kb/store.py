"""Vector store: ChromaDB persistent collection embedded with BGE-small via FastEmbed.

FastEmbed is ONNX-based (onnxruntime), so there is no torch dependency. The same
model embeds both documents and queries, which is required for sane retrieval.

Memory posture:
  - The embedder is loaded lazily on first call, never at import time.
  - Batched embedding keeps peak memory low during KB ingestion.
  - A manual drop() releases the embedder after a big ingest so the process
    can return to its baseline footprint.
"""
from __future__ import annotations

import gc
from functools import lru_cache
from typing import List, Optional

from config import settings


# Lazy singletons. Nothing touches fastembed or chromadb at import time.
_EMBEDDER = None
_COLLECTION = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from fastembed import TextEmbedding
        _EMBEDDER = TextEmbedding(model_name=settings.embed_model)
    return _EMBEDDER


def drop_embedder() -> None:
    """Release the embedder so the process returns to baseline memory.
    Safe to call after a big ingest; it will reload on the next query."""
    global _EMBEDDER
    _EMBEDDER = None
    gc.collect()


def embed(texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """Embed in small batches to keep peak memory low."""
    if not texts:
        return []
    out: List[List[float]] = []
    emb = _embedder()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for vec in emb.embed(batch):
            out.append(vec.tolist())
    return out


def _collection_impl():
    global _COLLECTION
    if _COLLECTION is None:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_dir)
        _COLLECTION = client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _COLLECTION


def _collection():
    return _collection_impl()


def reset_collection() -> None:
    global _COLLECTION
    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    _COLLECTION = None


def add(ids: List[str], documents: List[str], metadatas: List[dict], batch_size: int = 8) -> None:
    """Insert chunks in small batches so memory stays flat during a big ingest."""
    if not documents:
        return
    col = _collection()
    for i in range(0, len(documents), batch_size):
        sl = slice(i, i + batch_size)
        batch_docs = documents[sl]
        batch_ids = ids[sl]
        batch_metas = metadatas[sl]
        batch_embs = embed(batch_docs, batch_size=batch_size)
        col.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embs,
        )
        # Help the allocator reclaim the batch's intermediates between inserts.
        gc.collect()


def count() -> int:
    return _collection().count()


def query(
    text: str,
    *,
    stage: Optional[str] = None,
    chunk_type: Optional[str] = None,
    top_k: Optional[int] = None,
) -> List[str]:
    """Filtered retrieval.

    chunk_type: when set, filters to only chunks with that metadata type
                (e.g. 'service', 'pricing', 'faq', 'objection').
                Used for intent-based routing so the right slice of the KB
                is always pulled for the right question type.
    stage:      when set, also filters by stage (plus stage='any').
    """
    k = top_k or settings.retrieval_top_k

    where_clauses = []
    if stage:
        where_clauses.append(
            {"$or": [{"stage": {"$eq": stage}}, {"stage": {"$eq": "any"}}]}
        )
    if chunk_type:
        where_clauses.append({"type": {"$eq": chunk_type}})

    if len(where_clauses) == 0:
        where = None
    elif len(where_clauses) == 1:
        where = where_clauses[0]
    else:
        where = {"$and": where_clauses}

    res = _collection().query(
        query_embeddings=embed([text]),
        n_results=k,
        where=where,
    )
    docs = res.get("documents") or [[]]
    return docs[0] if docs else []


def query_by_type(chunk_type: str, top_k: int = 8) -> List[str]:
    """Pull all chunks of a given type directly (all service chunks, all pricing chunks)."""
    try:
        res = _collection().get(where={"type": {"$eq": chunk_type}})
        docs = res.get("documents") or []
        return docs[:top_k]
    except Exception:
        return []


def query_all_grounding() -> List[str]:
    """All chunks, for snapshot-time grounding context."""
    res = _collection().get()
    return res.get("documents") or []