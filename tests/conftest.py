"""Shared pytest fixtures for Insight-RAG tests."""

import os
import sys
import shutil
import tempfile
import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture()
def sample_chunks():
    """Return a small set of document chunks for testing."""
    return [
        {"text": "Machine learning is a subset of artificial intelligence.", "filename": "ml.txt", "chunk_index": 0},
        {"text": "Neural networks consist of layers of interconnected nodes.", "filename": "ml.txt", "chunk_index": 1},
        {"text": "The agreement may be terminated with 30 days written notice.", "filename": "contract.txt", "chunk_index": 0},
        {"text": "Confidential information shall not be disclosed to third parties.", "filename": "nda.txt", "chunk_index": 0},
        {"text": "Python is a high-level programming language used for data science.", "filename": "python.txt", "chunk_index": 0},
    ]


@pytest.fixture()
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB and clean up after test."""
    tmpdir = tempfile.mkdtemp(prefix="insightrag_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)
