"""
Retrieval Module
Top-k retrieval with reranking capabilities
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    """Document retrieval system"""
    
    def __init__(self, vector_store, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant document chunks"""
        k = top_k or self.top_k
        
        logger.info(f"Retrieving top {k} chunks for query: {query[:50]}...")
        
        results = self.vector_store.search(query, top_k=k)
        
        # Convert cosine distance → similarity.
        # ChromaDB cosine distance is in [0, 2]: 0 = identical, 2 = opposite.
        # similarity = 1 - distance maps that to [1, -1]; clamping to [0, 1]
        # keeps scores in a sensible range (negative only for near-opposites).
        for result in results:
            if 'distance' in result:
                similarity = max(0.0, min(1.0, 1.0 - result['distance']))
                result['similarity'] = similarity
                result['score'] = similarity
        
        logger.info(f"Retrieved {len(results)} chunks")
        
        return results
    
    def retrieve_with_threshold(self, query: str, similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """Retrieve chunks with minimum similarity threshold"""
        results = self.retrieve(query)
        
        # Filter by threshold
        filtered_results = [r for r in results if r.get('similarity', 0) >= similarity_threshold]
        
        if not filtered_results:
            return {
                'query': query,
                'results': [],
                'found': False,
                'message': 'No relevant documents found above similarity threshold'
            }
        
        return {
            'query': query,
            'results': filtered_results,
            'found': True,
            'top_score': filtered_results[0].get('similarity', 0)
        }
    
    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] {result.get('filename', 'Unknown')} (Chunk {result.get('chunk_index', 0)}):\n"
                f"{result.get('text', '')}"
            )
        
        return "\n\n".join(context_parts)
    
    def format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format sources for citation, deduplicating by (filename, chunk_index)"""
        sources = []
        seen = set()

        for result in results:
            filename = result.get('filename', 'Unknown')
            chunk_index = result.get('chunk_index', 0)
            key = (filename, chunk_index)
            if key in seen:
                continue
            seen.add(key)

            text = result.get('text', '')
            sources.append({
                'filename': filename,
                'chunk_index': chunk_index,
                'snippet': text[:200] + "..." if len(text) > 200 else text,
                'score': round(result.get('score', result.get('similarity', 0)), 4)
            })

        return sources


class Reranker:
    """Optional reranking for improved relevance"""
    
    def __init__(self):
        self.model = None
    
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank results based on additional criteria"""
        # Simple reranking based on:
        # 1. Original similarity score
        # 2. Text length (prefer substantial chunks)
        # 3. Position in document (prefer earlier chunks)
        
        for result in results:
            score = result.get('similarity', 0)
            
            # Boost score for chunks with substantial content
            text_length = len(result.get('text', ''))
            if text_length > 100:
                score *= 1.1
            
            # Small boost for earlier chunks (often more important)
            chunk_index = result.get('chunk_index', 0)
            if chunk_index < 3:
                score *= 1.05
            
            result['reranked_score'] = score
        
        # Sort by reranked score
        reranked = sorted(results, key=lambda x: x.get('reranked_score', 0), reverse=True)
        
        return reranked[:top_k]


def retrieve_documents(query: str, vector_store, top_k: int = 5) -> Dict[str, Any]:
    """Main retrieval function"""
    retriever = Retriever(vector_store, top_k=top_k)
    
    # Retrieve results
    results = retriever.retrieve(query, top_k=top_k)
    
    if not results:
        return {
            'query': query,
            'context': '',
            'sources': [],
            'found': False
        }
    
    # Build context
    context = retriever.build_context(results)
    
    # Format sources
    sources = retriever.format_sources(results)
    
    return {
        'query': query,
        'context': context,
        'sources': sources,
        'found': True,
        'top_score': results[0].get('similarity', 0) if results else 0
    }


if __name__ == "__main__":
    # Test retrieval
    from src.vector_store import create_vector_store
    
    print("Testing Retrieval...")
    vs = create_vector_store("docs")
    
    if vs.get_collection_stats()['total_chunks'] > 0:
        query = "What is the refund policy?"
        result = retrieve_documents(query, vs, top_k=3)
        print(f"\nQuery: {query}")
        print(f"Found: {result['found']}")
        print(f"Sources: {len(result['sources'])}")
    else:
        print("No documents in vector store. Add documents first.")
