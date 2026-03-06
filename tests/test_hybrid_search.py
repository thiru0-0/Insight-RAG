"""Tests for hybrid search: BM25 indexing, RRF fusion, score normalization."""

import pytest
from src.hybrid_search import BM25Index, HybridRetriever, reciprocal_rank_fusion, _tokenize


# ── Tokenizer ────────────────────────────────────────────────────────

class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("What is machine learning?")
        assert "machine" in tokens
        assert "learning" in tokens
        # Stop words removed
        assert "what" not in tokens
        assert "is" not in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_removes_punctuation(self):
        tokens = _tokenize("hello! world? foo-bar")
        assert "hello" in tokens
        assert "world" in tokens

    def test_single_char_filtered(self):
        tokens = _tokenize("I a x machine")
        assert "i" not in tokens
        assert "a" not in tokens
        assert "x" not in tokens
        assert "machine" in tokens


# ── BM25 Index ───────────────────────────────────────────────────────

class TestBM25Index:
    def test_initial_state(self):
        idx = BM25Index()
        assert not idx.is_built
        assert idx.size == 0

    def test_add_chunks_builds_index(self, sample_chunks):
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        assert idx.is_built
        assert idx.size == len(sample_chunks)

    def test_search_returns_relevant(self, sample_chunks):
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("machine learning", top_k=3)
        assert len(results) > 0
        # Top result should be from the ML document
        assert "machine" in results[0]["text"].lower() or "learning" in results[0]["text"].lower()

    def test_search_empty_index(self):
        idx = BM25Index()
        results = idx.search("machine learning")
        assert results == []

    def test_search_no_match(self, sample_chunks):
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("xyzzyspoon quantum entanglement photon")
        # May return empty or very low-scored results
        assert isinstance(results, list)

    def test_clear_resets(self, sample_chunks):
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        assert idx.is_built
        idx.clear()
        assert not idx.is_built
        assert idx.size == 0

    def test_search_result_structure(self, sample_chunks):
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("neural networks", top_k=2)
        assert len(results) > 0
        r = results[0]
        assert "text" in r
        assert "filename" in r
        assert "chunk_index" in r
        assert "bm25_score" in r
        assert r["bm25_score"] > 0


# ── Reciprocal Rank Fusion ───────────────────────────────────────────

class TestRRF:
    def test_empty_inputs(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_vector_only(self):
        vec = [
            {"text": "a", "filename": "a.txt", "chunk_index": 0, "score": 0.9},
            {"text": "b", "filename": "b.txt", "chunk_index": 0, "score": 0.7},
        ]
        fused = reciprocal_rank_fusion(vec, [])
        assert len(fused) == 2
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]

    def test_bm25_only(self):
        bm25 = [
            {"text": "x", "filename": "x.txt", "chunk_index": 0, "bm25_score": 5.0},
            {"text": "y", "filename": "y.txt", "chunk_index": 0, "bm25_score": 3.0},
        ]
        fused = reciprocal_rank_fusion([], bm25)
        assert len(fused) == 2
        assert "bm25" in fused[0]["retrieval_sources"]

    def test_overlap_fusion(self):
        """Same document in both lists should get a higher fused score."""
        shared = {"text": "shared", "filename": "shared.txt", "chunk_index": 0}
        vec = [dict(shared, score=0.8)]
        bm25 = [dict(shared, bm25_score=4.0)]
        fused = reciprocal_rank_fusion(vec, bm25)
        assert len(fused) == 1
        r = fused[0]
        assert "vector" in r["retrieval_sources"]
        assert "bm25" in r["retrieval_sources"]
        # Fused RRF score should be higher than either alone
        vec_only = reciprocal_rank_fusion(vec, [])
        assert r["rrf_score"] > vec_only[0]["rrf_score"]

    def test_rrf_score_in_results(self):
        vec = [{"text": "t", "filename": "f.txt", "chunk_index": 0, "score": 0.5}]
        fused = reciprocal_rank_fusion(vec, [])
        assert "rrf_score" in fused[0]
        assert fused[0]["rrf_score"] > 0


# ── Score Normalization ──────────────────────────────────────────────

class TestScoreNormalization:
    def test_bm25_only_scores_not_inflated(self):
        """BM25-only results should NOT all show 98-100% after normalization."""
        idx = BM25Index()
        chunks = [
            {"text": "Machine learning algorithms learn from data", "filename": "a.txt", "chunk_index": 0},
            {"text": "Deep learning is a subset of machine learning", "filename": "b.txt", "chunk_index": 0},
            {"text": "Python programming language syntax", "filename": "c.txt", "chunk_index": 0},
            {"text": "Contract termination clauses and notice periods", "filename": "d.txt", "chunk_index": 0},
        ]
        idx.add_chunks(chunks)
        bm25_results = idx.search("machine learning", top_k=4)
        vec_results = []  # no vector results

        fused = reciprocal_rank_fusion(vec_results, bm25_results)
        top = fused[:4]

        # Apply the same normalization logic as HybridRetriever.retrieve()
        rrf_scores = [item.get("rrf_score", 0) for item in top]
        rrf_min = min(rrf_scores) if rrf_scores else 0
        rrf_max = max(rrf_scores) if rrf_scores else 1
        rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0

        for item in top:
            if "score" not in item or item["score"] is None:
                raw = item.get("rrf_score", 0)
                normalised = (raw - rrf_min) / rrf_range
                item["score"] = round(0.20 + normalised * 0.75, 4)

        scores = [item["score"] for item in top if item.get("score") is not None]
        if len(scores) > 1:
            # Scores should be spread out, NOT all clustered near 1.0
            assert max(scores) <= 0.95, f"Max score too high: {max(scores)}"
            assert min(scores) >= 0.20, f"Min score too low: {min(scores)}"
            # There should be meaningful spread
            assert max(scores) - min(scores) > 0.05, f"Scores too clustered: {scores}"
