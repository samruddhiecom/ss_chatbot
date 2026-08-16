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
    # We supply embeddings explicitly, so no embedding_function is attached here.
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


def query(text: str, *, stage: Optional[str] = None, top_k: Optional[int] = None) -> List[str]:
    """Filtered retrieval. Pulls guidance for the current stage plus stage-agnostic
    company facts, so the generator only ever sees the right slice of the allowlist.
    """
    k = top_k or settings.retrieval_top_k
    where = None
    if stage:
        where = {"$or": [{"stage": {"$eq": stage}}, {"stage": {"$eq": "any"}}]}
    res = _collection().query(
        query_embeddings=embed([text]),
        n_results=k,
        where=where,
    )
    docs = res.get("documents") or [[]]
    return docs[0] if docs else []


def query_all_grounding() -> List[str]:
    """All A+B chunks, for snapshot-time grounding context."""
    res = _collection().get()
    return res.get("documents") or []
