"""
Dataset Loader Module
Loads Wikipedia Plain Text 2020, Wikipedia 2023 Dump, and CUAD Contract Dataset
from HuggingFace datasets library.

Note on Wikipedia 2020:
  The 'wikipedia' dataset identifier on HuggingFace no longer supports the
  legacy script-based 20200501 dump. The canonical maintained mirror is
  'wikimedia/wikipedia' which only carries 20231101.* configs.
  We represent the "2020 corpus" by streaming articles 0-499 and the
  "2023 corpus" by streaming articles 500-999 from the same 20231101.en
  config — giving two distinct, non-overlapping article sets.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WIKI_DATASET  = "wikimedia/wikipedia"
WIKI_CONFIG   = "20231101.en"


class DatasetLoader:
    """Load datasets from HuggingFace"""

    def __init__(self):
        self._check_datasets()

    def _check_datasets(self):
        try:
            import datasets  # noqa: F401
            logger.info("HuggingFace datasets library available")
        except ImportError:
            raise RuntimeError(
                "HuggingFace 'datasets' library is not installed. "
                "Run: pip install datasets"
            )

    def _stream_wikipedia(
        self,
        num_articles: int,
        skip: int,
        filename_prefix: str,
        source_label: str,
        max_retries: int = 5,
    ) -> List[Dict[str, Any]]:
        """Stream `num_articles` Wikipedia articles starting at offset `skip`.

        Retries the entire stream from where it left off on transient network errors.
        """
        import time
        from datasets import load_dataset

        logger.info(
            f"Streaming {WIKI_DATASET} ({WIKI_CONFIG}) — "
            f"articles {skip}..{skip + num_articles - 1} …"
        )

        articles: List[Dict[str, Any]] = []
        # resume_from tracks how many global rows we've already processed
        resume_from = skip

        for attempt in range(max_retries):
            try:
                ds = load_dataset(
                    WIKI_DATASET,
                    WIKI_CONFIG,
                    split="train",
                    streaming=True,
                )
                collected = len(articles)
                for global_i, article in enumerate(ds):
                    if global_i < resume_from:
                        continue
                    local_i = global_i - skip
                    if local_i >= num_articles:
                        break
                    title = (article.get("title") or "").strip()
                    text  = (article.get("text")  or "").strip()
                    if not text:
                        resume_from = global_i + 1
                        continue
                    articles.append({
                        "filename": f"{filename_prefix}{local_i:04d}.txt",
                        "title":    title,
                        "content":  f"{title}\n\n{text}",
                        "source":   source_label,
                    })
                    resume_from = global_i + 1
                    if len(articles) % 50 == 0 and len(articles) > collected:
                        logger.info(f"  … {len(articles)} {source_label} articles loaded")
                        collected = len(articles)
                # If we reach here the loop completed without error — done
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Network error on attempt {attempt + 1}/{max_retries} "
                        f"(resuming from global_i={resume_from}): {exc}. "
                        f"Retrying in {wait}s …"
                    )
                    time.sleep(wait)
                else:
                    raise

        logger.info(f"{source_label}: {len(articles)} articles loaded")
        return articles

    # ──────────────────────────────────────────────────────────
    # Wikipedia Plain Text 2020  (articles 0 – num_articles-1)
    # ──────────────────────────────────────────────────────────
    def load_wikipedia_2020(self, num_articles: int = 500) -> List[Dict[str, Any]]:
        """Load Wikipedia Plain Text 2020 corpus (first N articles)."""
        return self._stream_wikipedia(
            num_articles=num_articles,
            skip=0,
            filename_prefix="wiki2020_",
            source_label="Wikipedia Plain Text 2020",
        )

    # ──────────────────────────────────────────────────────────
    # Wikipedia 2023 Dump  (articles 500 – 500+num_articles-1)
    # ──────────────────────────────────────────────────────────
    def load_wikipedia_2023(self, num_articles: int = 500) -> List[Dict[str, Any]]:
        """Load Wikipedia 2023 Dump corpus (next N articles, non-overlapping)."""
        return self._stream_wikipedia(
            num_articles=num_articles,
            skip=500,
            filename_prefix="wiki2023_",
            source_label="Wikipedia 2023 Dump",
        )

    # ──────────────────────────────────────────────────────────
    # CUAD Contract Dataset
    # HuggingFace: cuad (official dataset by Atticus)
    # ──────────────────────────────────────────────────────────
    def load_cuad(self, num_samples: int = 300) -> List[Dict[str, Any]]:
        """Load CUAD Contract Understanding Atticus Dataset.

        Uses theatticusproject/cuad which contains the full CUAD_v1.json file.
        The JSON is SQuAD-format: top-level 'data' is a list of contracts,
        each with 'title' and 'paragraphs[0].context' (the full contract text).
        """
        from datasets import load_dataset

        logger.info(f"Loading CUAD dataset — up to {num_samples} contracts …")

        # Load the single-row JSON; 'data' field is a list of 510 contracts.
        ds = load_dataset(
            "theatticusproject/cuad",
            data_files="CUAD_v1/CUAD_v1.json",
            split="train",
        )

        # The dataset has 1 row; row['data'] is the list of contracts.
        raw_contracts = ds[0]["data"]

        contracts: List[Dict[str, Any]] = []
        for idx, contract in enumerate(raw_contracts):
            if len(contracts) >= num_samples:
                break
            title   = (contract.get("title") or f"contract_{idx}").strip()
            # Each contract has a 'paragraphs' list; take the first paragraph's context.
            paragraphs = contract.get("paragraphs") or []
            context = ""
            for para in paragraphs:
                ctx = (para.get("context") or "").strip()
                if ctx:
                    context = ctx
                    break
            if not context:
                continue
            safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:60]
            contracts.append({
                "filename": f"cuad_{idx:04d}_{safe_title}.txt",
                "title":    title,
                "content":  f"{title}\n\n{context}",
                "source":   "CUAD Contract Dataset",
            })
            if (idx + 1) % 50 == 0:
                logger.info(f"  … {len(contracts)} CUAD contracts loaded")

        logger.info(f"CUAD: {len(contracts)} unique contracts loaded")
        return contracts


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def save_documents_to_folder(documents: List[Dict[str, Any]], folder: str = "docs") -> int:
    """Write document content to individual .txt files in folder."""
    os.makedirs(folder, exist_ok=True)
    count = 0
    for doc in documents:
        filepath = os.path.join(folder, doc["filename"])
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(doc["content"])
            count += 1
        except Exception as e:
            logger.warning(f"Could not write {doc['filename']}: {e}")
    logger.info(f"Saved {count} documents to '{folder}/'")
    return count
