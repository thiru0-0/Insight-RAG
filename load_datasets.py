"""
Dataset Download + Ingest Pipeline
Downloads Wikipedia 2020, Wikipedia 2023, and CUAD from HuggingFace,
saves them to docs/, clears ChromaDB, and re-indexes everything.

Usage:
    python load_datasets.py
"""

import os
import sys
import shutil
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DOCS_DIR     = PROJECT_ROOT / "docs"
CHROMA_DIR   = PROJECT_ROOT / "data" / "chroma_db"


# ─────────────────────────────────────────────
# Step 1 – Clear existing data
# ─────────────────────────────────────────────
def _force_remove(path: Path):
    """Remove a directory tree, retrying with chmod on Windows permission errors."""
    import stat

    def _on_error(func, fpath, exc_info):
        # Make read-only files writable and retry
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass  # Best-effort; log later

    if path.exists():
        shutil.rmtree(path, onerror=_on_error)


def clear_data():
    logger.info("=" * 60)
    logger.info("Step 1 — Clearing existing docs/ and chroma_db/")
    logger.info("=" * 60)

    # Clear docs/
    if DOCS_DIR.exists():
        _force_remove(DOCS_DIR)
        logger.info(f"Deleted {DOCS_DIR}")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear chroma_db/ — may be locked on Windows; skip if still locked
    if CHROMA_DIR.exists():
        _force_remove(CHROMA_DIR)
        if CHROMA_DIR.exists():
            logger.warning(
                "chroma_db/ is locked by another process and could not be fully deleted. "
                "The ChromaDB collection will be cleared programmatically instead."
            )
        else:
            logger.info(f"Deleted {CHROMA_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Clear step done.")


# ─────────────────────────────────────────────
# Step 2 – Download datasets and save to docs/
# ─────────────────────────────────────────────
def download_datasets():
    logger.info("=" * 60)
    logger.info("Step 2 — Downloading datasets from HuggingFace")
    logger.info("=" * 60)

    # Add project root to path so src.* imports work
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.dataset_loader import DatasetLoader, save_documents_to_folder

    loader = DatasetLoader()
    all_docs = []

    # Wikipedia Plain Text 2020
    logger.info("--- Wikipedia Plain Text 2020 ---")
    try:
        docs = loader.load_wikipedia_2020(num_articles=500)
        all_docs.extend(docs)
        logger.info(f"Wikipedia 2020: {len(docs)} articles ready")
    except Exception as e:
        logger.error(f"Wikipedia 2020 failed: {e}")
        sys.exit(1)

    # Wikipedia 2023 Dump
    logger.info("--- Wikipedia 2023 Dump ---")
    try:
        docs = loader.load_wikipedia_2023(num_articles=500)
        all_docs.extend(docs)
        logger.info(f"Wikipedia 2023: {len(docs)} articles ready")
    except Exception as e:
        logger.error(f"Wikipedia 2023 failed: {e}")
        sys.exit(1)

    # CUAD Contract Dataset
    logger.info("--- CUAD Contract Dataset ---")
    try:
        docs = loader.load_cuad(num_samples=300)
        all_docs.extend(docs)
        logger.info(f"CUAD: {len(docs)} contracts ready")
    except Exception as e:
        logger.error(f"CUAD failed: {e}")
        sys.exit(1)

    logger.info(f"Total documents downloaded: {len(all_docs)}")

    # Save all to docs/
    saved = save_documents_to_folder(all_docs, str(DOCS_DIR))
    logger.info(f"Saved {saved} files to {DOCS_DIR}/")
    return saved


# ─────────────────────────────────────────────
# Step 3 – Chunk and index into ChromaDB
# ─────────────────────────────────────────────
def build_vector_store():
    logger.info("=" * 60)
    logger.info("Step 3 — Chunking and indexing into ChromaDB")
    logger.info("=" * 60)

    from src.ingest import DocumentLoader, TextChunker
    from src.vector_store import VectorStore

    chunk_size    = int(os.getenv("CHUNK_SIZE",    "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

    logger.info(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    # Load all saved docs
    loader    = DocumentLoader()
    documents = loader.load_folder(str(DOCS_DIR))
    logger.info(f"Loaded {len(documents)} documents from {DOCS_DIR}/")

    if not documents:
        logger.error("No documents found — aborting.")
        sys.exit(1)

    # Chunk
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks  = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # Index — clear existing collection first then add fresh chunks
    vs = VectorStore(
        persist_directory=str(CHROMA_DIR),
        collection_name="document_qa",
    )
    # Clear any existing data before indexing
    try:
        vs.clear()
        logger.info("Existing ChromaDB collection cleared")
    except Exception as e:
        logger.warning(f"Could not clear collection (may be empty): {e}")

    vs.add_chunks(chunks)

    stats = vs.get_collection_stats()
    logger.info(f"ChromaDB now contains {stats['total_chunks']} chunks")
    return stats["total_chunks"]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("Insight-RAG — Dataset Pipeline")
    logger.info("=" * 60)

    clear_data()
    saved  = download_datasets()
    chunks = build_vector_store()

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Documents saved : {saved}")
    logger.info(f"  Chunks indexed  : {chunks}")
    logger.info("Now restart the server:  python -m uvicorn src.main:app --host 0.0.0.0 --port 8012")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
