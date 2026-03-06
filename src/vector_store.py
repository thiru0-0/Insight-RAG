"""
Vector Store Module
Handles embeddings and Chroma vector database
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import uuid

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.model.encode([query])[0]


class VectorStore:
    """Chroma-based vector store for document retrieval"""
    
    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "document_qa"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_generator = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Chroma client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            logger.info("Initializing Chroma vector store")
            
            # Initialize Chroma client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize embedding function
            self.embedding_generator = EmbeddingGenerator()
            
            # Get or create collection
            # Use cosine distance so that similarity = 1 - distance is in [0, 1]
            COLLECTION_METADATA = {
                "hnsw:space": "cosine",
                "description": "Insight-RAG collection",
            }
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=COLLECTION_METADATA,
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 2000) -> bool:
        """Add document chunks to vector store in batches to avoid ChromaDB limits."""
        try:
            if not chunks:
                logger.warning("No chunks to add")
                return False
            
            logger.info(f"Adding {len(chunks)} chunks to vector store (batch_size={batch_size})")
            
            total_added = 0
            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start: batch_start + batch_size]

                texts = [chunk['text'] for chunk in batch]
                ids = [f"chunk_{uuid.uuid4().hex}" for _ in batch]
                metadatas = [
                    {
                        'filename': chunk.get('filename', ''),
                        'chunk_index': chunk.get('chunk_index', 0)
                    }
                    for chunk in batch
                ]

                # Generate embeddings for this batch
                embeddings = self.embedding_generator.embed_texts(texts)

                # Add batch to collection
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas
                )

                total_added += len(batch)
                logger.info(f"  Indexed {total_added}/{len(chunks)} chunks …")

            logger.info(f"Successfully added {total_added} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.embed_query(query)
            
            # Search in Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'text': doc,
                        'filename': results['metadatas'][0][i].get('filename', ''),
                        'chunk_index': results['metadatas'][0][i].get('chunk_index', 0),
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_chunks': 0, 'collection_name': self.collection_name}
    
    def clear(self) -> bool:
        """Clear all data from collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "description": "Insight-RAG collection"},
            )
            logger.info("Vector store cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False


def create_vector_store(docs_folder: str = "docs", chunk_size: int = 500, 
                        chunk_overlap: int = 50, persist_directory: str = "./data/chroma_db") -> VectorStore:
    """Create and populate vector store from documents"""
    
    # Import here to avoid circular imports
    from src.ingest import ingest_documents
    
    # Ingest documents
    chunks = ingest_documents(docs_folder, chunk_size, chunk_overlap)
    
    if not chunks:
        logger.warning("No chunks generated. Creating empty vector store.")
    
    # Create vector store
    vector_store = VectorStore(persist_directory=persist_directory)
    
    # Add chunks
    if chunks:
        vector_store.add_chunks(chunks)
    
    return vector_store


if __name__ == "__main__":
    # Test vector store
    print("Testing Vector Store...")
    vs = create_vector_store("docs")
    stats = vs.get_collection_stats()
    print(f"Collection stats: {stats}")
