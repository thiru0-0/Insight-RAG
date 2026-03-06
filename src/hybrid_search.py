"""
Hybrid Search Module
Combines vector (semantic) search with BM25 (keyword) search
using Reciprocal Rank Fusion (RRF) for improved retrieval accuracy.
"""

import re
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ── Stop words for tokenization ──────────────────────────────────────
_STOP_WORDS = frozenset({
    "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "with", "by", "what", "which", "how", "when", "where", "who", "why",
    "can", "do", "does", "did", "are", "was", "were", "be", "from", "this",
    "that", "it", "as", "at", "about", "not", "but", "if", "has", "have",
    "had", "will", "would", "could", "should", "may", "might", "been",
    "being", "its", "than", "then", "so", "such", "there", "their", "they",
})


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in tokens if len(t) > 1 and t not in _STOP_WORDS]


class BM25Index:
    """In-memory BM25 keyword index over document chunks."""

    def __init__(self):
        self._corpus_tokens: List[List[str]] = []
        self._corpus_meta: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def size(self) -> int:
        return len(self._corpus_tokens)

    # ── Build ────────────────────────────────────────────────────────
    def build_from_chromadb(self, collection, batch_size: int = 5000) -> None:
        """
        Pull every document from a ChromaDB collection and build the
        BM25 index.  This is called once at startup.
        """
        total = collection.count()
        if total == 0:
            logger.warning("BM25: collection is empty, nothing to index")
            return

        logger.info(f"BM25: indexing {total} chunks from ChromaDB …")

        self._corpus_tokens = []
        self._corpus_meta = []

        for offset in range(0, total, batch_size):
            batch = collection.get(
                offset=offset,
                limit=batch_size,
                include=["documents", "metadatas"],
            )
            docs = batch.get("documents", [])
            metas = batch.get("metadatas", [])
            ids = batch.get("ids", [])

            for i, doc in enumerate(docs):
                tokens = _tokenize(doc)
                self._corpus_tokens.append(tokens)
                meta = metas[i] if i < len(metas) else {}
                self._corpus_meta.append({
                    "id": ids[i] if i < len(ids) else "",
                    "text": doc,
                    "filename": meta.get("filename", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                })

        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._built = True
        logger.info(f"BM25: index built — {len(self._corpus_tokens)} chunks")

    # ── Search ───────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Return top-k results scored by BM25."""
        if not self._built or self._bm25 is None:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                break  # remaining are zero/negative
            meta = self._corpus_meta[idx]
            results.append({
                "text": meta["text"],
                "filename": meta["filename"],
                "chunk_index": meta["chunk_index"],
                "bm25_score": score,
            })

        return results

    # ── Incremental update ───────────────────────────────────────────
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add new chunks and rebuild the BM25 model."""
        for chunk in chunks:
            tokens = _tokenize(chunk.get("text", ""))
            self._corpus_tokens.append(tokens)
            self._corpus_meta.append({
                "id": "",
                "text": chunk.get("text", ""),
                "filename": chunk.get("filename", ""),
                "chunk_index": chunk.get("chunk_index", 0),
            })

        # Rebuild BM25 (fast — pure numpy)
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
            self._built = True

    def clear(self) -> None:
        """Reset the index."""
        self._corpus_tokens = []
        self._corpus_meta = []
        self._bm25 = None
        self._built = False


# ── Reciprocal Rank Fusion ───────────────────────────────────────────
def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = sum( weight / (k + rank) )

    Higher k smooths the ranking; 60 is the standard default.
    Returns a unified list sorted by fused score.
    """
    # Build a dict keyed by (filename, chunk_index) → best record + rrf score
    fused: Dict[tuple, Dict[str, Any]] = {}

    for rank, item in enumerate(vector_results, start=1):
        key = (item.get("filename", ""), item.get("chunk_index", 0))
        entry = fused.get(key, {"record": item, "rrf": 0.0, "sources": []})
        entry["rrf"] += vector_weight / (k + rank)
        entry["sources"].append("vector")
        entry["record"] = item  # prefer vector record (has distance/score)
        fused[key] = entry

    for rank, item in enumerate(bm25_results, start=1):
        key = (item.get("filename", ""), item.get("chunk_index", 0))
        entry = fused.get(key, {"record": item, "rrf": 0.0, "sources": []})
        entry["rrf"] += bm25_weight / (k + rank)
        if "bm25" not in entry["sources"]:
            entry["sources"].append("bm25")
        if "vector" not in entry["sources"]:
            entry["record"] = item  # only bm25 had this chunk
        fused[key] = entry

    # Sort by RRF score descending
    ranked = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)

    results = []
    for entry in ranked:
        record = dict(entry["record"])
        record["rrf_score"] = round(entry["rrf"], 6)
        record["retrieval_sources"] = entry["sources"]
        results.append(record)

    return results


class HybridRetriever:
    """
    Drop-in replacement for the basic Retriever that combines
    vector search (ChromaDB) with keyword search (BM25) using RRF.
    """

    def __init__(self, vector_store, bm25_index: BM25Index, top_k: int = 5):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Hybrid retrieve: vector + BM25 → RRF → top_k."""
        k = top_k or self.top_k

        # 1. Vector search — fetch more candidates than final k for fusion
        vector_candidates = self.vector_store.search(query, top_k=k * 3)

        # Attach similarity scores
        for item in vector_candidates:
            if "distance" in item:
                sim = max(0.0, min(1.0, 1.0 - item["distance"]))
                item["similarity"] = sim
                item["score"] = sim

        # 2. BM25 search
        bm25_candidates = self.bm25_index.search(query, top_k=k * 3)

        # 3. Fuse with RRF
        fused = reciprocal_rank_fusion(vector_candidates, bm25_candidates)

        # 4. Keep top_k
        top_results = fused[:k]

        # Ensure every result has a valid score (use existing score or derive from RRF)
        # For items that only came from BM25 (no vector similarity score),
        # use min-max normalization of RRF scores across the result set.
        rrf_scores = [item.get("rrf_score", 0) for item in top_results]
        rrf_min = min(rrf_scores) if rrf_scores else 0
        rrf_max = max(rrf_scores) if rrf_scores else 1
        rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0

        for item in top_results:
            if "score" not in item or item["score"] is None:
                # Min-max normalize the RRF score into [0.20, 0.95]
                # so BM25-only results get sensible display scores
                raw = item.get("rrf_score", 0)
                normalised = (raw - rrf_min) / rrf_range  # [0, 1]
                item["score"] = round(0.20 + normalised * 0.75, 4)  # [0.20, 0.95]
                item["similarity"] = item["score"]

        logger.info(
            f"Hybrid search: {len(vector_candidates)} vector + "
            f"{len(bm25_candidates)} BM25 → {len(top_results)} fused results"
        )

        return top_results

    # ── Delegate utility methods to stay compatible ──────────────────
    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.get('filename', 'Unknown')} "
                f"(Chunk {r.get('chunk_index', 0)}):\n"
                f"{r.get('text', '')}"
            )
        return "\n\n".join(parts)

    def format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format sources for citation, deduplicating by (filename, chunk_index)."""
        sources = []
        seen = set()
        for r in results:
            filename = r.get("filename", "Unknown")
            chunk_index = r.get("chunk_index", 0)
            key = (filename, chunk_index)
            if key in seen:
                continue
            seen.add(key)
            text = r.get("text", "")
            sources.append({
                "filename": filename,
                "chunk_index": chunk_index,
                "snippet": text[:200] + "..." if len(text) > 200 else text,
                "score": round(r.get("score", r.get("similarity", 0)), 4),
                "retrieval_sources": r.get("retrieval_sources", ["vector"]),
            })
        return sources
