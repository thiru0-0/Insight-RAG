"""Integration smoke test for frontend/backend app flow."""

import os
import shutil
import tempfile
from pathlib import Path


def run_smoke_test() -> None:
    temp_db = tempfile.mkdtemp(prefix="qa_smoke_")
    docs_dir = Path(__file__).parent / "docs"
    try:
        os.environ["CHROMA_PERSIST_DIRECTORY"] = temp_db

        from fastapi.testclient import TestClient
        from src.main import app

        with TestClient(app) as client:
            app_res = client.get("/app")
            assert app_res.status_code == 200, "GET /app failed"
            html = app_res.text
            for element_id in ["askBtn", "clearBtn", "uploadBtn", "refreshBtn", "clearIndexBtn"]:
                assert element_id in html, f"UI element missing: {element_id}"

            health_res = client.get("/health")
            assert health_res.status_code == 200, "GET /health failed"
            health_data = health_res.json()
            assert health_data.get("status") == "healthy", "Health status is not healthy"

            query_res = client.post(
                "/query",
                json={
                    "question": "What is the refund policy?",
                    "top_k": 3,
                    "use_citations": True,
                },
            )
            assert query_res.status_code == 200, "POST /query failed"
            query_data = query_res.json()
            for key in ["answer", "sources", "confidence", "query"]:
                assert key in query_data, f"Missing key in query response: {key}"

            ingest_content = b"The release notes mention dark theme support for all users."
            ingest_res = client.post(
                "/ingest",
                files={"file": ("smoke_test_release_notes.txt", ingest_content, "text/plain")},
            )
            assert ingest_res.status_code == 200, "POST /ingest failed"
            ingest_data = ingest_res.json()
            assert ingest_data.get("status") == "success", "Ingest did not return success"

            stats_res = client.get("/stats")
            assert stats_res.status_code == 200, "GET /stats failed"

            samples_res = client.get("/samples")
            assert samples_res.status_code == 200, "GET /samples failed"
            samples_data = samples_res.json()
            assert isinstance(samples_data.get("samples"), list), "Samples payload is invalid"
            assert isinstance(samples_data.get("datasets"), dict), "Dataset status payload is invalid"

            clear_res = client.post("/clear")
            assert clear_res.status_code == 200, "POST /clear failed"
            clear_data = clear_res.json()
            assert clear_data.get("status") == "success", "Clear did not return success"
    finally:
        for generated_file in docs_dir.glob("*_smoke_test_release_notes.txt"):
            try:
                generated_file.unlink()
            except OSError:
                pass
        os.environ.pop("CHROMA_PERSIST_DIRECTORY", None)
        shutil.rmtree(temp_db, ignore_errors=True)

    print("Smoke test passed: frontend and backend flow is functional.")


if __name__ == "__main__":
    run_smoke_test()
