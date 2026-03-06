"""API integration tests using FastAPI TestClient."""

import os
import sys
import shutil
import tempfile
from pathlib import Path
import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with temporary ChromaDB and docs directories.

    Uses a tiny single-file docs folder so bootstrap completes in seconds
    instead of re-indexing the full 1,301-file dataset.
    """
    temp_db = tempfile.mkdtemp(prefix="insightrag_api_test_")
    temp_docs = tempfile.mkdtemp(prefix="insightrag_docs_test_")

    # Write a small test document
    test_doc = Path(temp_docs) / "test_doc.txt"
    test_doc.write_text(
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn from data. The refund policy allows returns within "
        "30 days of purchase for a full refund. Neural networks consist of "
        "layers of interconnected nodes used for pattern recognition."
    )

    os.environ["CHROMA_PERSIST_DIRECTORY"] = temp_db

    # Patch DOCS_DIR before importing the app so bootstrap uses our tiny dir
    import src.main as main_module
    original_docs_dir = main_module.DOCS_DIR
    main_module.DOCS_DIR = Path(temp_docs)

    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as c:
        yield c

    # Restore
    main_module.DOCS_DIR = original_docs_dir
    os.environ.pop("CHROMA_PERSIST_DIRECTORY", None)
    shutil.rmtree(temp_db, ignore_errors=True)
    shutil.rmtree(temp_docs, ignore_errors=True)


# ── Health & Info ────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "vector_store_stats" in data

    def test_root_returns_info(self, client):
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert "endpoints" in data

    def test_stats_returns_200(self, client):
        res = client.get("/stats")
        assert res.status_code == 200
        data = res.json()
        assert "total_chunks" in data
        assert "retrieval_method" in data

    def test_samples_returns_200(self, client):
        res = client.get("/samples")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data["samples"], list)
        assert isinstance(data["datasets"], dict)


# ── Frontend ─────────────────────────────────────────────────────────

class TestFrontend:
    def test_app_ui_serves_html(self, client):
        res = client.get("/app")
        assert res.status_code == 200
        html = res.text
        # Check for key UI elements
        for element_id in ["askBtn", "clearBtn", "uploadBtn"]:
            assert element_id in html, f"UI element missing: {element_id}"


# ── Ingest ───────────────────────────────────────────────────────────

class TestIngest:
    def test_ingest_txt_file(self, client):
        content = b"The refund policy allows returns within 30 days of purchase for a full refund."
        res = client.post(
            "/ingest",
            files={"file": ("test_policy.txt", content, "text/plain")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["chunks_added"] > 0

    def test_ingest_rejects_unsupported_type(self, client):
        res = client.post(
            "/ingest",
            files={"file": ("test.jpg", b"fake image data", "image/jpeg")},
        )
        assert res.status_code == 400

    def test_ingest_rejects_large_file(self, client):
        # 11 MB of data
        big_content = b"x" * (11 * 1024 * 1024)
        res = client.post(
            "/ingest",
            files={"file": ("big.txt", big_content, "text/plain")},
        )
        assert res.status_code == 400
        assert "too large" in res.json()["detail"].lower()


# ── Query ────────────────────────────────────────────────────────────

class TestQuery:
    def test_query_returns_answer(self, client):
        res = client.post(
            "/query",
            json={"question": "What is the refund policy?", "top_k": 3, "use_citations": True},
        )
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "query" in data
        assert "retrieval_method" in data

    def test_query_empty_question_rejected(self, client):
        res = client.post("/query", json={"question": "", "top_k": 3})
        assert res.status_code in (400, 422)  # validation error

    def test_query_returns_session_id(self, client):
        res = client.post(
            "/query",
            json={"question": "What is machine learning?", "top_k": 3},
        )
        assert res.status_code == 200
        data = res.json()
        assert "session_id" in data
        assert data["session_id"] is not None

    def test_query_with_session_continuity(self, client):
        # First query — creates session
        r1 = client.post("/query", json={"question": "What is AI?", "top_k": 3})
        sid = r1.json()["session_id"]

        # Second query — same session
        r2 = client.post(
            "/query",
            json={"question": "Tell me more about it", "top_k": 3, "session_id": sid},
        )
        assert r2.status_code == 200
        assert r2.json()["session_id"] == sid

    def test_query_fallback_on_irrelevant(self, client):
        res = client.post(
            "/query",
            json={"question": "xyzzyspoon quantum entanglement baryogenesis", "top_k": 3},
        )
        assert res.status_code == 200
        data = res.json()
        # Should return the mandatory fallback
        assert "could not find" in data["answer"].lower() or len(data["sources"]) == 0


# ── Sessions ─────────────────────────────────────────────────────────

class TestSessions:
    def test_create_session(self, client):
        res = client.post("/session")
        assert res.status_code == 200
        data = res.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 12

    def test_get_session_history(self, client):
        # Create session and add a turn via query
        r1 = client.post("/query", json={"question": "What is AI?", "top_k": 3})
        sid = r1.json()["session_id"]

        res = client.get(f"/session/{sid}/history")
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == sid
        assert "turns" in data
        assert "count" in data

    def test_delete_session(self, client):
        r1 = client.post("/session")
        sid = r1.json()["session_id"]

        res = client.delete(f"/session/{sid}")
        assert res.status_code == 200
        assert res.json()["status"] == "cleared"


# ── Clear ────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_vector_store(self, client):
        res = client.post("/clear")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
