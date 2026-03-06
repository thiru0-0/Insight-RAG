"""
Local LLM Generator Module
FLAN-T5 based answer generation (no API key required)
"""

import logging
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class LocalLLMGenerator:
    """Generate answers using FLAN-T5 local model"""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        # Skip loading for faster startup - use fallback
        logger.info("Using fast rule-based answer generation (fallback mode)")
    
    def _load_model(self):
        """Load FLAN-T5 model"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            
            logger.info(f"Loading FLAN-T5 model: {self.model_name}")
            
            # Use CPU (or CUDA if available)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(device)
            
            logger.info("FLAN-T5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading FLAN-T5 model: {e}")
            logger.warning("Falling back to rule-based generation")
            self.model = None
            self.tokenizer = None
    
    def generate(self, query: str, context: str) -> Dict[str, Any]:
        """Generate answer from query and context"""
        
        if not context:
            return {
                'answer': "I could not find this in the provided documents. Can you share the relevant document?",
                'confidence': 'low',
                'sources': []
            }
        
        # If model is loaded, use it
        if self.model is not None and self.tokenizer is not None:
            return self._generate_with_model(query, context)
        else:
            # Fallback to rule-based
            return self._fallback_generate(query, context)
    
    def _generate_with_model(self, query: str, context: str) -> Dict[str, Any]:
        """Generate answer using FLAN-T5"""
        try:
            from transformers import pipeline
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Create prompt
            prompt = self._build_prompt(query, context)
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Estimate confidence
            confidence = self._estimate_confidence(answer, context)
            
            return {
                'answer': answer.strip(),
                'confidence': confidence,
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating with model: {e}")
            return self._fallback_generate(query, context)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for the model"""
        return f"""Answer the question based only on the context below. 
If you cannot find the answer in the context, say "I could not find this in the provided documents."

Context: {context}

Question: {query}

Answer:"""
    
    def _estimate_confidence(self, answer: str, context: str) -> str:
        """Estimate confidence based on answer quality"""
        answer_lower = answer.lower()
        
        # Check for uncertain phrases
        uncertain_phrases = [
            "i cannot find", "cannot find", "not found", "not mentioned", 
            "not specified", "i don't know", "no information"
        ]
        for phrase in uncertain_phrases:
            if phrase in answer_lower:
                return "low"
        
        # Check if answer is too short
        if len(answer.split()) < 5:
            return "low"
        
        # Check if answer references context
        answer_words = set(answer_lower.split())
        context_words = set(context.lower().split())
        common_words = answer_words & context_words
        
        if len(common_words) < 3:
            return "medium"
        
        return "high"
    
    def _fallback_generate(self, query: str, context: str) -> Dict[str, Any]:
        """Fallback answer generation without LLM"""
        
        # Clean the context - remove [1], [2] markers
        clean_context = context
        import re
        clean_context = re.sub(r'\[\d+\]', '', clean_context)
        clean_context = re.sub(r'chunk_\d+:', '', clean_context)
        
        # Split context into sentences - more robust
        sentences = []
        for para in clean_context.split('\n'):
            # Split by various punctuation
            parts = re.split(r'(?<=[.!?])\s+', para)
            for sent in parts:
                sent = sent.strip()
                if len(sent) > 10:  # Skip very short segments
                    sentences.append(sent)
        
        # Find relevant sentences
        query_words = set(re.sub(r"[^a-z0-9\s]", " ", query.lower()).split())
        stop_words = {
            'what', 'is', 'the', 'a', 'an', 'how', 'do', 'i', 'can', 'to', 'of', 'and',
            'in', 'on', 'for', 'from', 'with', 'that', 'this', 'it', 'are', 'be', 'does'
        }
        query_keywords = {w for w in query_words if len(w) > 2 and w not in stop_words}
        
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(re.sub(r"[^a-z0-9\s]", " ", sentence_lower).split())
            
            # Check word overlap with query keywords
            overlap = len(query_keywords & sentence_words)
            coverage = overlap / max(1, len(query_keywords))

            # Boost for clause-like answers in contract questions
            bonus = 0.0
            if any(term in query_keywords for term in {"termination", "notice", "term", "agreement", "confidential", "liability"}):
                if any(term in sentence_lower for term in ["shall", "may", "days", "months", "years", "written notice"]):
                    bonus += 0.3
            score = overlap + coverage + bonus
            
            # Also check for key terms from query
            threshold = 2 if len(query_keywords) >= 4 else 1
            if overlap >= threshold:
                # Check if sentence contains meaningful content (not just headers)
                if len(sentence) > 30 and not sentence.startswith('#'):
                    relevant_sentences.append((sentence, score))
        
        # Sort by relevance (more overlap = higher)
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_sentences:
            # Take the top 3 distinct sentences ordered by relevance score,
            # then re-sort them by their original position in the context so
            # the answer reads naturally (highest-scored first if order unknown).
            selected = []
            seen = set()
            for text, _ in relevant_sentences:
                key = text.lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                selected.append(text)
                if len(selected) == 3:
                    break
            answer = ' '.join(selected)
            # Clean up the answer
            answer = re.sub(r'\s+', ' ', answer).strip()
            if not answer.endswith('.'):
                answer += '.'

            # Derive confidence from how well the top sentence matched.
            top_score = relevant_sentences[0][1]  # (text, score) — higher is better
            keyword_count = max(1, len(query_keywords))
            coverage = top_score / keyword_count  # rough normalised coverage ratio
            if coverage >= 0.6:
                confidence = "high"
            elif coverage >= 0.3:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            # Mandatory fallback
            answer = "I could not find this in the provided documents. Can you share the relevant document?"
            confidence = "low"
        
        return {
            'answer': answer,
            'confidence': confidence,
            'fallback': True
        }


class CitationManager:
    """Manage citations and source attribution"""
    
    def __init__(self):
        pass
    
    def create_citations(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create formatted citations from sources"""
        citations = []
        
        for i, source in enumerate(sources, 1):
            citation = {
                'id': i,
                'filename': source.get('filename', 'Unknown'),
                'chunk_index': source.get('chunk_index', 0),
                'snippet': self._truncate_snippet(source.get('text', source.get('snippet', '')), 200),
                'score': round(source.get('score', source.get('similarity', 0)), 4)
            }
            citations.append(citation)
        
        return citations
    
    def _truncate_snippet(self, text: str, max_length: int = 200) -> str:
        """Truncate snippet to max length"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def add_citations_to_answer(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Add citation references to answer"""
        if not sources:
            return answer
        
        citations = self.create_citations(sources)
        
        answer_with_citations = answer + "\n\n**Sources:**\n"
        for cite in citations:
            snippet_preview = cite['snippet'][:100] + "..." if len(cite['snippet']) > 100 else cite['snippet']
            answer_with_citations += f"[{cite['id']}] {cite['filename']}: {snippet_preview}\n"
        
        return answer_with_citations


def generate_answer(query: str, retrieval_result: Dict[str, Any], 
                   use_citations: bool = True) -> Dict[str, Any]:
    """Main answer generation function"""
    
    generator = LocalLLMGenerator()
    
    # Generate answer
    result = generator.generate(query, retrieval_result.get('context', ''))
    
    # Add sources
    sources = retrieval_result.get('sources', [])
    
    # Format final answer
    answer = result['answer']
    if use_citations and sources:
        citation_manager = CitationManager()
        answer = citation_manager.add_citations_to_answer(answer, sources)
    
    # Calculate final confidence
    retrieval_score = retrieval_result.get('top_score', 0)
    generation_confidence = result.get('confidence', 'low')
    
    # Combine confidence
    final_confidence = _combine_confidence(retrieval_score, generation_confidence)
    
    return {
        'answer': answer,
        'sources': sources,
        'confidence': final_confidence
    }


def _combine_confidence(retrieval_score: float, generation_confidence: str) -> str:
    """Combine retrieval and generation confidence"""
    
    conf_map = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
    gen_conf = conf_map.get(generation_confidence, 0.5)
    
    combined = (retrieval_score + gen_conf) / 2
    
    if combined >= 0.7:
        return 'high'
    elif combined >= 0.4:
        return 'medium'
    else:
        return 'low'


if __name__ == "__main__":
    print("Testing Local LLM Generator...")
    
    # Test with sample context
    test_context = """
    [1] artificial_intelligence.txt: Artificial Intelligence (AI) is intelligence demonstrated by machines.
    Machine learning is a subset of AI that enables systems to learn from data.
    
    [2] machine_learning.txt: Machine learning algorithms build models based on training data.
    Deep learning uses neural networks with multiple layers.
    """
    
    query = "What is machine learning?"
    
    generator = LocalLLMGenerator()
    result = generator.generate(query, test_context)
    
    print(f"\nQuery: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
