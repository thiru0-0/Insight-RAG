"""
Insight-RAG API
FastAPI application for RAG-based question answering
"""

import os
import logging
from typing import Optional, List, Tuple, Any, Dict
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
vector_store = None
retriever = None
generator = None
reranker = None
bm25_index = None
chat_memory = None
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

MANDATORY_FALLBACK = "I could not find this in the provided documents. Can you share the relevant document?"

DATASET_SAMPLE_QA = [
    {
        "source": "Wikipedia 2020",
        "question": "What is machine learning?",
        "answer": "Machine learning is a part of AI where systems learn from data to make predictions or decisions without explicit rule-by-rule programming.",
    },
    {
        "source": "Wikipedia 2023",
        "question": "What does natural language processing do?",
        "answer": "NLP helps computers process and understand human language in text or speech.",
    },
    {
        "source": "CUAD Contract",
        "question": "What is the termination notice period in the service agreement?",
        "answer": "Either party may terminate the agreement with thirty (30) days written notice.",
    },
    {
        "source": "CUAD Contract",
        "question": "How long does the sample NDA term remain in effect?",
        "answer": "The NDA remains in effect for two (2) years from the effective date.",
    },
]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: Optional[int] = Field(default=5, ge=1, le=10, description="Number of results to retrieve")
    use_citations: Optional[bool] = Field(default=True, description="Include citations in response")
    session_id: Optional[str] = Field(default=None, description="Chat session ID for conversation memory")


class SummarizeRequest(BaseModel):
    filename: str = Field(..., min_length=1, description="Document filename to summarize")
    max_sentences: Optional[int] = Field(default=7, ge=3, le=20, description="Max sentences in summary")


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: str
    query: str
    session_id: Optional[str] = None
    query_rewrite: Optional[dict] = None
    retrieval_method: str = "hybrid"


class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    documents_processed: int


class HealthResponse(BaseModel):
    status: str
    vector_store_stats: dict


def _keyword_tokens(text: str) -> set:
    tokens = [t.strip(".,:;!?()[]{}\"'`").lower() for t in text.split()]
    stop = {
        "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
        "what", "which", "how", "when", "where", "who", "why", "can", "do", "does", "did",
        "are", "was", "were", "be", "from", "this", "that", "it", "as", "at", "about"
    }
    return {t for t in tokens if len(t) > 2 and t not in stop}


def _is_relevant(question: str, retrieval_result: List[Dict[str, Any]]) -> bool:
    if not retrieval_result:
        return False

    query_tokens = _keyword_tokens(question)
    if not query_tokens:
        return True

    combined_text = " ".join(item.get("text", "")[:400] for item in retrieval_result[:3])
    doc_tokens = _keyword_tokens(combined_text)
    overlap = len(query_tokens & doc_tokens)
    if overlap >= 1:
        return True

    top_score = retrieval_result[0].get("score", retrieval_result[0].get("similarity", 0.0))
    min_score = float(os.getenv("MIN_RELEVANCE_SCORE", "0.30"))
    return top_score >= min_score


def _fallback_response(question: str, session_id: Optional[str] = None) -> QueryResponse:
    return QueryResponse(
        answer=MANDATORY_FALLBACK,
        sources=[],
        confidence="low",
        query=question,
        session_id=session_id,
        retrieval_method="hybrid",
    )


def _dataset_status() -> Dict[str, Any]:
    docs = (
        list(DOCS_DIR.glob("*.txt"))
        + list(DOCS_DIR.glob("*.md"))
        + list(DOCS_DIR.glob("*.pdf"))
    )
    status = {
        "wikipedia_2020_docs": 0,
        "wikipedia_2023_docs": 0,
        "cuad_docs": 0,
        "other_docs": 0,
    }
    for doc in docs:
        name = doc.name.lower()
        if name.startswith("wiki2020_"):
            status["wikipedia_2020_docs"] += 1
        elif name.startswith("wiki2023_"):
            status["wikipedia_2023_docs"] += 1
        elif name.startswith("cuad_"):
            status["cuad_docs"] += 1
        else:
            status["other_docs"] += 1
    status["total_docs"] = len(docs)
    return status


def initialize_system():
    """Initialize the RAG system"""
    global vector_store, retriever, generator, reranker, bm25_index, chat_memory
    
    logger.info("Initializing Insight-RAG System...")
    
    # Import components - use local imports
    import sys
    
    # Add project root to path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from src.vector_store import VectorStore
    from src.retriever import Reranker
    from src.llm_generator import LocalLLMGenerator
    from src.hybrid_search import BM25Index, HybridRetriever
    from src.query_engine import ChatMemory
    
    # Initialize vector store - use same directory as setup
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", str(PROJECT_ROOT / "data" / "chroma_db"))
    collection_name = "document_qa"  # Fixed name to use existing collection
    
    logger.info(f"Using persist_directory: {persist_dir}")
    
    vector_store = VectorStore(persist_directory=persist_dir, collection_name=collection_name)

    # Bootstrap collection from docs folder if empty
    stats = vector_store.get_collection_stats()
    if stats.get("total_chunks", 0) == 0:
        logger.info("Collection is empty. Bootstrapping from docs folder...")
        from src.ingest import DocumentLoader, TextChunker

        docs_dir = str(DOCS_DIR)
        loader = DocumentLoader()
        documents = loader.load_folder(docs_dir)

        if documents:
            chunker = TextChunker(
                chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            )
            chunks = chunker.chunk_documents(documents)
            if chunks:
                vector_store.add_chunks(chunks)
                logger.info(f"Bootstrapped {len(chunks)} chunks from docs folder")
        else:
            logger.warning(f"No documents found in {docs_dir}")
    
    # ── Build BM25 keyword index ────────────────────────────────────
    bm25_index = BM25Index()
    bm25_index.build_from_chromadb(vector_store.collection)
    logger.info(f"BM25 index ready: {bm25_index.size} chunks indexed")

    # ── Initialize hybrid retriever (vector + BM25 + reranker) ──────
    top_k = int(os.getenv("TOP_K", "5"))
    retriever = HybridRetriever(vector_store, bm25_index, top_k=top_k)
    
    # ── Initialize chat memory ──────────────────────────────────────
    chat_memory = ChatMemory()
    logger.info("Chat memory initialized")

    # Initialize generator
    generator = LocalLLMGenerator()
    
    # Initialize reranker (singleton — shared across requests)
    reranker = Reranker()
    logger.info("Reranker initialized (singleton)")
    
    logger.info("System initialized successfully (hybrid search + chat memory enabled)")


def ensure_system_ready() -> Tuple[Any, Any, Any]:
    if vector_store is None or retriever is None or generator is None or bm25_index is None or reranker is None:
        raise HTTPException(status_code=503, detail="System is not initialized yet")
    return vector_store, retriever, generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system on startup, clean up on shutdown"""
    initialize_system()
    yield


app = FastAPI(
    title="Insight-RAG",
    description="RAG-based Question Answering System with Citations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Insight-RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "app": "/app",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "summarize": "/summarize (POST)",
            "documents": "/documents (GET)",
            "ingest": "/ingest (POST)",
            "stats": "/stats (GET)"
        }
    }


@app.get("/app", tags=["UI"])
async def app_ui():
    """Serve mobile-first frontend UI"""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(ui_path)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        stats = vector_store.get_collection_stats() if vector_store else {'total_chunks': 0}
        return HealthResponse(
            status="healthy",
            vector_store_stats=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get system statistics"""
    try:
        stats = vector_store.get_collection_stats() if vector_store else {}
        chunk_count = stats.get('total_chunks', 0)
        dataset_info = _dataset_status()
        return {
            "total_chunks": chunk_count,
            "total_documents": dataset_info.get("total_docs", 0),
            "collection_name": stats.get('collection_name', 'N/A'),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "llm_model": os.getenv("LLM_MODEL", "rule-based extractor"),
            "retrieval_method": "hybrid (vector + BM25 + reranker)",
            "bm25_indexed": bm25_index.size if bm25_index else 0,
            "dataset_status": dataset_info,
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(
    file: UploadFile = File(..., description="Document to ingest (.txt, .md, .pdf)"),
):
    """
    Ingest a single document into the vector store
    """
    try:
        current_vector_store, _, _ = ensure_system_ready()

        # Validate file type
        allowed_extensions = ['.txt', '.md', '.pdf']
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {allowed_extensions}"
            )
        
        # Read file content
        content = await file.read()

        # Enforce file size limit (10 MB)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10 MB)")
        
        safe_name = Path(file.filename).name

        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = DOCS_DIR / safe_name

        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process only this document
        from src.ingest import DocumentLoader, TextChunker

        loader = DocumentLoader()
        content_text = loader.load_document(str(temp_path))

        if not content_text.strip():
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if file_ext == ".pdf":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Could not extract text from the PDF. "
                        "The file may be scanned/image-based or encrypted. "
                        "Please upload a searchable PDF or a .txt/.md file."
                    ),
                )
            raise HTTPException(status_code=400, detail="Could not extract text from document")

        # Chunk documents
        chunker = TextChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
        )
        chunks = chunker.chunk_text(content_text, safe_name)

        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks generated from document")
        
        # Add to vector store
        added = current_vector_store.add_chunks(chunks)
        if not added:
            raise HTTPException(status_code=500, detail="Failed to store document chunks in vector database")

        # Update BM25 index incrementally
        if bm25_index is not None:
            bm25_index.add_chunks(chunks)
            logger.info(f"BM25 index updated: +{len(chunks)} chunks")
        
        logger.info(f"Ingested {safe_name}: {len(chunks)} chunks")
        
        return IngestResponse(
            status="success",
            chunks_added=len(chunks),
            documents_processed=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/folder", response_model=IngestResponse, tags=["Documents"])
async def ingest_folder(folder_path: str = Form(..., description="Path to folder with documents")):
    """
    Ingest all documents from a folder
    """
    try:
        current_vector_store, _, _ = ensure_system_ready()

        if not os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")
        
        from src.ingest import DocumentLoader, TextChunker
        
        loader = DocumentLoader()
        documents = loader.load_folder(folder_path)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents found in folder")
        
        chunker = TextChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
        )
        chunks = chunker.chunk_documents(documents)
        
        current_vector_store.add_chunks(chunks)

        # Update BM25 index incrementally
        if bm25_index is not None:
            bm25_index.add_chunks(chunks)
            logger.info(f"BM25 index updated: +{len(chunks)} chunks")
        
        logger.info(f"Ingested folder {folder_path}: {len(chunks)} chunks from {len(documents)} docs")
        
        return IngestResponse(
            status="success",
            chunks_added=len(chunks),
            documents_processed=len(documents)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Ask a question and get an answer with citations.
    Supports hybrid search (vector + BM25), query rewriting, and chat memory.
    """
    try:
        _, current_retriever, current_generator = ensure_system_ready()

        from src.query_engine import rewrite_query

        # Validate input
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # ── Session management ──────────────────────────────────────
        session_id = request.session_id
        history = []
        if chat_memory is not None:
            if not session_id:
                session_id = chat_memory.create_session()
            history = chat_memory.get_history(session_id)

        # ── Query rewriting (coreference + synonym expansion) ───────
        rewrite_result = rewrite_query(
            query=request.question,
            history=history if history else None,
            expand_synonyms=True,
        )
        search_query = rewrite_result["rewritten"]
        logger.info(f"Query: '{request.question[:50]}' → '{search_query[:60]}'")

        # ── Hybrid retrieval ────────────────────────────────────────
        top_k = request.top_k if request.top_k is not None else 5
        retrieval_result = current_retriever.retrieve(search_query, top_k=top_k)

        scored_results = [
            item for item in retrieval_result
            if float(item.get("score", item.get("similarity", 0.0)) or 0.0) > 0.0
        ]

        if not scored_results:
            return _fallback_response(request.question, session_id)

        if not _is_relevant(request.question, scored_results):
            return _fallback_response(request.question, session_id)

        # ── Rerank for multi-document reasoning ─────────────────────
        scored_results = reranker.rerank(request.question, scored_results, top_k=top_k)

        # Build context
        context = current_retriever.build_context(scored_results)
        
        # Generate answer
        answer_result = current_generator.generate(request.question, context)
        
        # Format sources
        sources = current_retriever.format_sources(scored_results) if request.use_citations else []

        # ── Store turn in chat memory ───────────────────────────────
        if chat_memory is not None and session_id:
            chat_memory.add_turn(session_id, request.question, answer_result["answer"])

        return QueryResponse(
            answer=answer_result['answer'],
            sources=sources,
            confidence=answer_result['confidence'],
            query=request.question,
            session_id=session_id,
            query_rewrite=rewrite_result if rewrite_result["was_rewritten"] else None,
            retrieval_method="hybrid",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear", tags=["Admin"])
async def clear_vector_store():
    """Clear all documents from vector store"""
    try:
        current_vector_store, _, _ = ensure_system_ready()
        cleared = current_vector_store.clear()
        if not cleared:
            raise HTTPException(status_code=500, detail="Failed to clear vector store")
        # Also clear BM25 index
        if bm25_index is not None:
            bm25_index.clear()
            logger.info("BM25 index cleared")
        return {"status": "success", "message": "Vector store and BM25 index cleared"}
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """List all unique document filenames in the vector store."""
    try:
        ensure_system_ready()
        # Fetch all metadatas from the collection
        result = vector_store.collection.get(include=["metadatas"])
        filenames = set()
        for meta in (result.get("metadatas") or []):
            fn = (meta or {}).get("filename", "")
            if fn:
                filenames.add(fn)
        sorted_names = sorted(filenames)
        return {"documents": sorted_names, "count": len(sorted_names)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", tags=["Query"])
async def summarize_document(request: SummarizeRequest):
    """
    Generate an extractive summary of a specific document.
    Retrieves all chunks for the document from ChromaDB and selects
    the most representative sentences using a TextRank-inspired approach.
    """
    import re
    import math

    try:
        ensure_system_ready()

        filename = request.filename.strip()
        max_sentences = request.max_sentences or 7

        # ── Retrieve all chunks for this document from ChromaDB ─────
        result = vector_store.collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
        )

        docs = result.get("documents") or []
        if not docs:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for document: {filename}",
            )

        # ── Re-assemble document text in chunk_index order ──────────
        metas = result.get("metadatas") or []
        paired = list(zip(docs, metas))
        paired.sort(key=lambda x: int(x[1].get("chunk_index", 0) or 0))
        full_text = "\n".join(chunk_text for chunk_text, _ in paired)

        # ── Split into sentences ────────────────────────────────────
        raw_sentences = re.split(r'(?<=[.!?])\s+', full_text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 20]

        if not sentences:
            return {
                "filename": filename,
                "summary": full_text[:500],
                "total_chunks": len(docs),
                "total_sentences": 0,
                "sentences_selected": 0,
            }

        # ── Tokenise helper ─────────────────────────────────────────
        STOP = {
            "the", "and", "for", "are", "but", "not", "you", "all", "any",
            "can", "had", "her", "was", "one", "our", "out", "day", "get",
            "has", "him", "his", "how", "its", "new", "now", "old", "see",
            "two", "way", "who", "did", "she", "too", "use", "that", "this",
            "with", "they", "have", "from", "been", "were", "said", "each",
            "which", "their", "when", "will", "more", "than", "also", "into",
            "some", "what", "there", "about", "would", "could", "should",
            "these", "those", "such", "other", "been", "being", "very",
            "just", "only", "then", "much", "both", "them", "same",
        }

        def tokenise(text):
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            return [w for w in words if w not in STOP]

        # ── Build word frequency table (TF) ─────────────────────────
        word_freq = {}
        for sent in sentences:
            for w in tokenise(sent):
                word_freq[w] = word_freq.get(w, 0) + 1

        if not word_freq:
            # No meaningful words — just return first few sentences
            selected = sentences[:max_sentences]
            return {
                "filename": filename,
                "summary": " ".join(selected),
                "total_chunks": len(docs),
                "total_sentences": len(sentences),
                "sentences_selected": len(selected),
            }

        max_freq = max(word_freq.values())
        for w in word_freq:
            word_freq[w] /= max_freq  # normalise to [0, 1]

        # ── Score each sentence ─────────────────────────────────────
        scored = []
        for idx, sent in enumerate(sentences):
            tokens = tokenise(sent)
            if not tokens:
                scored.append((idx, sent, 0.0))
                continue
            tf_score = sum(word_freq.get(w, 0) for w in tokens) / len(tokens)

            # Positional boost: first & last sentences are more important
            position_boost = 0.0
            if idx < 3:
                position_boost = 0.15 * (3 - idx) / 3
            elif idx >= len(sentences) - 2:
                position_boost = 0.05

            # Length normalisation: prefer medium-length sentences
            length_penalty = 0.0
            if len(tokens) < 5:
                length_penalty = -0.1
            elif len(tokens) > 40:
                length_penalty = -0.05

            final_score = tf_score + position_boost + length_penalty
            scored.append((idx, sent, final_score))

        # ── Select top sentences, then re-order by position ─────────
        scored.sort(key=lambda x: x[2], reverse=True)
        selected_indices = sorted([s[0] for s in scored[:max_sentences]])
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = " ".join(summary_sentences)

        # Clean up
        summary = re.sub(r'\s+', ' ', summary).strip()

        return {
            "filename": filename,
            "summary": summary,
            "total_chunks": len(docs),
            "total_sentences": len(sentences),
            "sentences_selected": len(summary_sentences),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/samples", tags=["Info"])
async def get_dataset_samples():
    return {
        "datasets": _dataset_status(),
        "samples": DATASET_SAMPLE_QA,
    }


@app.post("/session", tags=["Chat"])
async def create_session():
    """Create a new chat session for conversation memory."""
    if chat_memory is None:
        raise HTTPException(status_code=503, detail="Chat memory not initialized")
    session_id = chat_memory.create_session()
    return {"session_id": session_id}


@app.delete("/session/{session_id}", tags=["Chat"])
async def delete_session(session_id: str):
    """Clear a chat session."""
    if chat_memory is None:
        raise HTTPException(status_code=503, detail="Chat memory not initialized")
    chat_memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/session/{session_id}/history", tags=["Chat"])
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    if chat_memory is None:
        raise HTTPException(status_code=503, detail="Chat memory not initialized")
    history = chat_memory.get_history(session_id)
    return {"session_id": session_id, "turns": history, "count": len(history)}


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
