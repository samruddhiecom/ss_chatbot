"""Vector store: ChromaDB persistent collection embedded with BGE-small via FastEmbed.

FastEmbed is ONNX-based (onnxruntime), so there is no torch dependency. The same
model embeds both documents and queries, which is required for sane retrieval.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from config import settings


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=settings.embed_model)


def embed(texts: List[str]) -> List[List[float]]:
    return [vec.tolist() for vec in _embedder().embed(texts)]


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    _collection.cache_clear()


def add(ids: List[str], documents: List[str], metadatas: List[dict]) -> None:
    if not documents:
        return
    _collection().add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embed(documents),
    )


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