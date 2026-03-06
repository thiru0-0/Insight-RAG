# Insight-RAG — Hybrid RAG Document Q&A

Production-grade Document Q&A system built for the AI & Programming Hackathon.
Uses **hybrid retrieval** (vector search + BM25 keyword search) with Reciprocal Rank Fusion for accurate, grounded answers from indexed documents.

## Features

- **Hybrid Search** — combines semantic vector search (ChromaDB) with keyword search (BM25) using Reciprocal Rank Fusion (RRF) for superior retrieval accuracy
- **Query Rewriting** — synonym expansion and coreference resolution using conversation history
- **Chat Memory** — server-side session management with conversation context carryover
- **Heuristic Reranker** — re-scores retrieval results for multi-document reasoning
- **Grounding Check** — keyword-overlap + score-threshold validation ensures answers come from indexed documents
- **Mandatory Fallback** — returns `"I could not find this in the provided documents. Can you share the relevant document?"` when no relevant content is found
- **Evidence Citations** — every response includes `filename`, `snippet`, `score`, and `retrieval_sources`
- **Confidence Labels** — `high`, `medium`, `low` based on retrieval coverage
- **File Upload** — ingest `.txt`, `.md`, `.pdf` files directly from the UI (max 10 MB)
- **Mobile-first Frontend** — dark purple UI served at `/app`

## Architecture

```
User Question
    │
    ▼
Query Rewriter (synonym expansion + coreference resolution)
    │
    ▼
┌───────────────────┐     ┌──────────────────┐
│ Vector Search     │     │ BM25 Keyword     │
│ (ChromaDB cosine) │     │ Search (in-mem)  │
└───────────────────┘     └──────────────────┘
         \                      /
          ▼                    ▼
     Reciprocal Rank Fusion (RRF)
              │
              ▼
       Heuristic Reranker
              │
              ▼
     Grounding Check (keyword overlap + min score)
              │
              ▼
     Rule-based Answer Generator
              │
              ▼
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
| Document parser | PyPDF2 + text readers |
| Frontend | Vanilla HTML/CSS/JS (mobile-first) |

## Project Structure

```
Insight-RAG/
  docs/                        # Indexed documents (1,301 .txt files)
  data/chroma_db/              # Persistent ChromaDB (67,934 chunks)
  src/
    main.py                    # FastAPI app, routes, lifespan init
    hybrid_search.py           # BM25Index, HybridRetriever, RRF fusion
    query_engine.py            # ChatMemory, query rewriting, synonym expansion
    retriever.py               # Retriever, Reranker classes
    vector_store.py            # EmbeddingGenerator, VectorStore (ChromaDB)
    llm_generator.py           # Rule-based grounded answer generator
    ingest.py                  # DocumentLoader, TextChunker
    dataset_loader.py          # Wikipedia/CUAD dataset loading
    static/index.html          # Dark purple frontend UI
  tests/
    conftest.py                # Shared pytest fixtures
    test_hybrid_search.py      # BM25, RRF, normalization tests (14 tests)
    test_query_engine.py       # Synonym, coreference, memory tests (25 tests)
    test_api.py                # API endpoint integration tests (17 tests)
  load_datasets.py             # Dataset load + index helper
  run_pipeline.py              # Dataset loading entry point
  test_frontend_backend_flow.py  # Integration smoke test
  requirements.txt
  .env.example
  frontend-design.md           # UI design methodology (DFII scoring)
```

## Setup

```bash
pip install -r requirements.txt
python load_datasets.py
```

`load_datasets.py` will:
1. Load available dataset content (Wikipedia 2020/2023, CUAD contracts)
2. Save docs into `docs/`
3. Chunk, embed, and index into ChromaDB

If the collection is empty at server startup, it auto-bootstraps from `docs/`.

## Run the Server

```bash
python src/main.py
```

Or with a custom port:

```bash
set API_PORT=8012 && python src/main.py
```

Then open:
- **Frontend UI:** `http://127.0.0.1:8000/app`
- **Swagger docs:** `http://127.0.0.1:8000/docs`
- **Health check:** `http://127.0.0.1:8000/health`

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

Or run the integration smoke test directly:

```bash
python test_frontend_backend_flow.py
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
  "query_rewrite": {
    "original": "What is the termination notice period?",
    "rewritten": "What is the termination notice period? terminate end",
    "display_query": "What is the termination notice period?",
    "expanded_terms": ["terminate", "end"],
    "was_rewritten": true,
    "reason": "Expanded with synonym terms"
  },
  "retrieval_method": "hybrid"
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

## Key Design Decisions

- **No paid API keys** — the generator is rule-based (extracts relevant sentences from retrieved context). No OpenAI/Anthropic dependency.
- **Hybrid retrieval** — vector search alone misses keyword-exact matches; BM25 alone misses semantic similarity. RRF fusion combines both ranked lists.
- **Min-max score normalization** — BM25-only results get display scores in [0.20, 0.95] via min-max normalization of RRF scores, preventing inflated 98-100% scores.
- **Singleton reranker** — `Reranker` is instantiated once at startup, not per-request.
- **Server-side sessions** — chat memory is stored server-side (10 turns/session, 1hr TTL, 200 max sessions) for coreference resolution.
- **Grounding check** — queries are validated against retrieved content using keyword overlap and minimum relevance score (`MIN_RELEVANCE_SCORE=0.30`).

## Hackathon Guideline Compliance

- **Correctness & functionality (40%):** grounded retrieval + mandatory fallback + citations + confidence
- **AI/ML quality (30%):** embeddings + hybrid vector/BM25 search + RRF fusion + reranker + query rewriting
- **API design & engineering (20%):** structured endpoints + validation + error handling + session management
- **Documentation (10%):** this README + runnable commands + API examples + 59-test suite

## Important Behavior

- No fabricated sources or hallucinated evidence
- Graceful handling for empty/invalid input
- If indexed content is empty at startup, app auto-bootstraps from `docs/`
- All answers are strictly derived from retrieved document chunks
