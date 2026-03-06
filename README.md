# Insight-RAG — Hybrid RAG Document Q&A

Production-grade Document Q&A system built for the AI & Programming Hackathon.
Uses **hybrid retrieval** (vector search + BM25 keyword search) with Reciprocal Rank Fusion for accurate, grounded answers from indexed documents.

## Features

- **Hybrid Search** — combines semantic vector search (ChromaDB) with keyword search (BM25) using Reciprocal Rank Fusion (RRF) for superior retrieval accuracy
- **Document Summarization** — extractive summarization of any indexed document using TF-based sentence scoring with positional boosting
- **Query Rewriting** — synonym expansion and coreference resolution using conversation history
- **Chat Memory** — server-side session management with conversation context carryover
- **Heuristic Reranker** — re-scores retrieval results for multi-document reasoning
- **Grounding Check** — keyword-overlap + score-threshold validation ensures answers come from indexed documents
- **Mandatory Fallback** — returns `"I could not find this in the provided documents. Can you share the relevant document?"` when no relevant content is found
- **Evidence Citations** — every response includes `filename`, `snippet`, `score`, and `retrieval_sources`
- **Confidence Labels** — `high`, `medium`, `low` based on retrieval coverage
- **File Upload** — ingest `.txt`, `.md`, `.pdf` files directly from the UI (max 10 MB)
- **Mobile-first Frontend** — dark purple UI served at `/app`

## Quick Start (Clone & Run)

### Prerequisites

- **Python 3.10+** (tested with 3.12)
- **pip** (Python package manager)
- **Git**
- ~2 GB disk space (for dependencies + embeddings model + document index)

### Step 1: Clone the Repository

```bash
git clone https://github.com/thiru0-0/Insight-RAG.git
cd Insight-RAG
```

### Step 2: Create a Virtual Environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, ChromaDB, sentence-transformers, PyTorch, rank_bm25, and all other dependencies.

### Step 4: Configure Environment (Optional)

```bash
# Copy the example config
cp .env.example .env

# Edit .env if you want to change defaults (port, chunk size, etc.)
```

Default settings work out of the box — no API keys required.

### Step 5: Start the Server

```bash
python src/main.py
```

On first startup, the system will:
1. Download the `all-MiniLM-L6-v2` embedding model (~90 MB, one-time download)
2. Auto-bootstrap from the `docs/` folder — chunk and embed all 1,301 documents (~67,934 chunks)
3. Build the BM25 keyword index in memory

> **Note:** First startup takes 5-15 minutes depending on hardware (embedding 67K chunks). Subsequent startups load from the persisted ChromaDB and take ~30 seconds.

### Step 6: Open the App

Once you see `Uvicorn running on http://0.0.0.0:8000`, open your browser:

| URL | Description |
|---|---|
| [http://localhost:8000/app](http://localhost:8000/app) | Frontend UI (dark purple interface) |
| [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger API documentation |
| [http://localhost:8000/health](http://localhost:8000/health) | Health check endpoint |

### Custom Port

```bash
# Linux / macOS
API_PORT=8012 python src/main.py

# Windows CMD
set API_PORT=8012 && python src/main.py

# Windows PowerShell
$env:API_PORT="8012"; python src/main.py
```

## Run Tests

```bash
pytest tests/ -v
```

**59 tests** across 3 test files:

| File | Tests | Coverage |
|---|---|---|
| `test_hybrid_search.py` | 14 | Tokenizer, BM25 index, RRF fusion, score normalization |
| `test_query_engine.py` | 25 | Pronouns, coreference, synonyms, ChatMemory sessions |
| `test_api.py` | 17 | Health, frontend, ingest, query, sessions, clear |

## Architecture

```
User Question
    |
    v
Query Rewriter (synonym expansion + coreference resolution)
    |
    v
+-------------------+     +------------------+
| Vector Search     |     | BM25 Keyword     |
| (ChromaDB cosine) |     | Search (in-mem)  |
+-------------------+     +------------------+
         \                      /
          v                    v
     Reciprocal Rank Fusion (RRF)
              |
              v
       Heuristic Reranker
              |
              v
     Grounding Check (keyword overlap + min score)
              |
              v
     Rule-based Answer Generator
              |
              v
     Response: answer + sources + confidence
```

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Vector store | ChromaDB (persistent, cosine metric) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Keyword search | BM25Okapi (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Generator | Local rule-based extractor (no paid API) |
| Summarizer | Extractive TF + positional scoring |
| Document parser | PyPDF2 + text readers |
| Frontend | Vanilla HTML/CSS/JS (mobile-first) |

## Project Structure

```
Insight-RAG/
  project_docs/                  # Hackathon submission materials (PPT, diagrams)
  docs/                          # Indexed documents (1,301 .txt files)
  data/chroma_db/                # Persistent ChromaDB (67,934 chunks) [auto-created]
  src/
    main.py                      # FastAPI app, routes, lifespan init
    hybrid_search.py             # BM25Index, HybridRetriever, RRF fusion
    query_engine.py              # ChatMemory, query rewriting, synonym expansion
    retriever.py                 # Retriever, Reranker classes
    vector_store.py              # EmbeddingGenerator, VectorStore (ChromaDB)
    llm_generator.py             # Rule-based grounded answer generator
    ingest.py                    # DocumentLoader, TextChunker
    dataset_loader.py            # Wikipedia/CUAD dataset loading
    static/index.html            # Dark purple frontend UI
  tests/
    conftest.py                  # Shared pytest fixtures
    test_hybrid_search.py        # BM25, RRF, normalization tests (14 tests)
    test_query_engine.py         # Synonym, coreference, memory tests (25 tests)
    test_api.py                  # API endpoint integration tests (17 tests)
  load_datasets.py               # Dataset load + index helper
  run_pipeline.py                # Dataset loading entry point
  requirements.txt               # Python dependencies
  .env.example                   # Environment variable template
  Dockerfile                     # Docker build for deployment
  render.yaml                    # Render.com deployment blueprint
  frontend-design.md             # UI design methodology (DFII scoring)
```

## API Endpoints

### Info & Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root info with endpoint listing |
| `GET` | `/health` | Service health + vector store stats |
| `GET` | `/stats` | Model, index, and dataset statistics |
| `GET` | `/samples` | Sample Q&A pairs and dataset status |
| `GET` | `/app` | Serve frontend UI |

### Query

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | Ask a grounded question with hybrid retrieval |

**Request:**

```json
{
  "question": "What is the termination notice period?",
  "top_k": 5,
  "use_citations": true,
  "session_id": "a1b2c3d4e5f6"
}
```

**Response:**

```json
{
  "answer": "Either party may terminate the agreement with thirty (30) days written notice.",
  "sources": [
    {
      "filename": "cuad_contract_001.txt",
      "chunk_index": 3,
      "snippet": "Either party may terminate the agreement...",
      "score": 0.8234,
      "retrieval_sources": ["vector", "bm25"]
    }
  ],
  "confidence": "high",
  "query": "What is the termination notice period?",
  "session_id": "a1b2c3d4e5f6",
  "retrieval_method": "hybrid"
}
```

### Summarization

| Method | Path | Description |
|---|---|---|
| `GET` | `/documents` | List all unique document filenames in the vector store |
| `POST` | `/summarize` | Generate an extractive summary of a specific document |

**Request:**

```json
{
  "filename": "wiki2020_artificial_intelligence.txt",
  "max_sentences": 7
}
```

**Response:**

```json
{
  "filename": "wiki2020_artificial_intelligence.txt",
  "summary": "Artificial Intelligence (AI) is intelligence demonstrated by machines...",
  "total_chunks": 12,
  "total_sentences": 45,
  "sentences_selected": 7
}
```

### Documents

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload and index a single file (`.txt`, `.md`, `.pdf`, max 10 MB) |
| `POST` | `/ingest/folder` | Index all supported files from a folder path |
| `POST` | `/clear` | Clear the vector store and BM25 index |

### Chat Sessions

| Method | Path | Description |
|---|---|---|
| `POST` | `/session` | Create a new chat session |
| `GET` | `/session/{id}/history` | Get conversation history for a session |
| `DELETE` | `/session/{id}` | Delete a chat session |

## Environment Variables

All variables are optional — defaults work out of the box:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model name |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma_db` | ChromaDB storage path |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Default number of retrieval results |
| `MIN_RELEVANCE_SCORE` | `0.30` | Minimum score for grounding check |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |

## Key Design Decisions

- **No paid API keys** — the generator is rule-based (extracts relevant sentences from retrieved context). No OpenAI/Anthropic dependency.
- **Hybrid retrieval** — vector search alone misses keyword-exact matches; BM25 alone misses semantic similarity. RRF fusion combines both ranked lists.
- **Min-max score normalization** — BM25-only results get display scores in [0.20, 0.95] via min-max normalization of RRF scores, preventing inflated 98-100% scores.
- **Singleton reranker** — `Reranker` is instantiated once at startup, not per-request.
- **Server-side sessions** — chat memory is stored server-side (10 turns/session, 1hr TTL, 200 max sessions) for coreference resolution.
- **Grounding check** — queries are validated against retrieved content using keyword overlap and minimum relevance score.
- **Extractive summarization** — no LLM needed; uses TF-based sentence scoring with positional boosting and length normalization to select representative sentences.

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Make sure venv is activated and `pip install -r requirements.txt` was run |
| First startup is slow | Normal — embedding 67K chunks takes 5-15 min. Subsequent starts use cached ChromaDB |
| Port already in use | Use `API_PORT=8012 python src/main.py` to pick a different port |
| `torch` install fails | Try `pip install torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only |
| Empty responses | Check `/health` endpoint; if `total_chunks: 0`, run `python load_datasets.py` |

## Hackathon Guideline Compliance

- **Correctness & functionality (40%):** grounded retrieval + mandatory fallback + citations + confidence + summarization
- **AI/ML quality (30%):** embeddings + hybrid vector/BM25 search + RRF fusion + reranker + query rewriting
- **API design & engineering (20%):** structured endpoints + validation + error handling + session management
- **Documentation (10%):** this README + runnable commands + API examples + 59-test suite

## Important Behavior

- No fabricated sources or hallucinated evidence
- Graceful handling for empty/invalid input
- If indexed content is empty at startup, app auto-bootstraps from `docs/`
- All answers are strictly derived from retrieved document chunks

## License

Built for the AI & Programming Hackathon by THIRUMURUGESH J D.
