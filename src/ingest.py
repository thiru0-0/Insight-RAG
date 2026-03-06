"""
Document Ingestion Module
Loads and chunks documents from various formats
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Load documents from various file formats"""
    
    @staticmethod
    def load_text(file_path: str) -> str:
        """Load .txt and .md files"""
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading text file {file_path}: {e}")
                return ""
        logger.error(f"Could not decode text file {file_path} with supported encodings")
        return ""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Load .pdf files using PyPDF2"""
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        logger.warning(f"PDF is encrypted and could not be decrypted: {file_path}")
                        return ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return ""
    
    def load_document(self, file_path: str) -> str:
        """Load document based on file extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.txt', '.md']:
            return self.load_text(file_path)
        elif ext == '.pdf':
            return self.load_pdf(file_path)
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return ""
    
    def load_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Load all supported documents from a folder"""
        documents = []
        
        supported_extensions = ['.txt', '.md', '.pdf']
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    file_path = os.path.join(root, file)
                    content = self.load_document(file_path)
                    
                    if content.strip():
                        documents.append({
                            'filename': file,
                            'path': file_path,
                            'content': content
                        })
                        logger.info(f"Loaded: {file}")
                    else:
                        logger.warning(f"Empty or unreadable: {file}")
        
        return documents


class TextChunker:
    """Split text into chunks for embedding"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, filename: str = "") -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        chunks = []
        
        if not text.strip():
            return chunks
        
        # Split by paragraphs first to preserve semantic meaning
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'filename': filename,
                    'chunk_index': len(chunks)
                })
                
                # Keep overlap for context
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
            else:
                current_chunk += para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'filename': filename,
                'chunk_index': len(chunks)
            })
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents"""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['content'], doc['filename'])
            all_chunks.extend(chunks)
            logger.info(f"Chunked {doc['filename']} into {len(chunks)} chunks")
        
        return all_chunks


def ingest_documents(docs_folder: str = "docs", chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """Main ingestion function"""
    logger.info(f"Starting ingestion from {docs_folder}")
    
    loader = DocumentLoader()
    documents = loader.load_folder(docs_folder)
    
    if not documents:
        logger.warning(f"No documents found in {docs_folder}")
        return []
    
    logger.info(f"Loaded {len(documents)} documents")
    
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(documents)
    
    logger.info(f"Created {len(chunks)} total chunks")
    
    return chunks


if __name__ == "__main__":
    # Test ingestion
    chunks = ingest_documents("docs")
    print(f"\nTotal chunks: {len(chunks)}")
    if chunks:
        print(f"\nSample chunk:")
        print(f"  File: {chunks[0]['filename']}")
        print(f"  Text: {chunks[0]['text'][:200]}...")
