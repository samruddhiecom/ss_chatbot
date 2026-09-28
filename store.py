"""Vector store: ChromaDB persistent collection embedded with BGE-small via FastEmbed.

FastEmbed is ONNX-based (onnxruntime), so there is no torch dependency. The same
model embeds both documents and queries, which is required for sane retrieval.

Reranking: flashrank cross-encoder (ms-marco-MiniLM-L-12-v2) re-scores the
candidate set after vector retrieval. Falls back to cosine-only if the model
has not been downloaded yet (first cold start on a fresh Railway deploy).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=settings.embed_model)


def embed(texts: List[str]) -> List[List[float]]:
    return [vec.tolist() for vec in _embedder().embed(texts)]


@lru_cache(maxsize=1)
def _ranker():
    """Load flashrank cross-encoder. Returns None on any failure so the caller
    can degrade gracefully to cosine-only ranking."""
    try:
        from flashrank import Ranker
        return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=settings.reranker_cache_dir)
    except Exception as e:
        logger.warning("Reranker unavailable (%s) — cosine-only retrieval active.", e)
        return None


def _rerank(query: str, docs: List[str], top_n: int) -> List[str]:
    """Cross-encode query against docs and return top_n by relevance score.
    Falls back to returning the first top_n docs (cosine order) if the ranker
    is not loaded."""
    if not docs:
        return docs
    ranker = _ranker()
    if ranker is None:
        return docs[:top_n]
    try:
        from flashrank import RerankRequest
        passages = [{"id": i, "text": d} for i, d in enumerate(docs)]
        req = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(req)
        # results is sorted by score descending already
        return [docs[r["id"]] for r in results[:top_n]]
    except Exception as e:
        logger.warning("Rerank failed (%s) — returning cosine order.", e)
        return docs[:top_n]


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
    top_k: Optional[int] = None,
    rerank: bool = True,
) -> List[str]:
    """Filtered retrieval with optional cross-encoder reranking.

    Fetches candidate_k candidates from Chroma (cosine), then reranks them
    with the cross-encoder and returns the top top_k results. The wider
    candidate pool means more context is considered before the reranker prunes.
    """
    top_k = top_k or settings.retrieval_top_k
    # Fetch a wider candidate pool for the reranker to work with
    candidate_k = min(top_k * 3, count() or top_k * 3)

    where = None
    if stage:
        where = {"$or": [{"stage": {"$eq": stage}}, {"stage": {"$eq": "any"}}]}

    res = _collection().query(
        query_embeddings=embed([text]),
        n_results=max(candidate_k, 1),
        where=where,
    )
    docs = (res.get("documents") or [[]])[0]

    if rerank and len(docs) > top_k:
        docs = _rerank(text, docs, top_k)
    else:
        docs = docs[:top_k]

    return docs


def query_all_grounding() -> List[str]:
    """All A+B chunks, for snapshot-time grounding context."""
    res = _collection().get()
    return res.get("documents") or []
