"""Document processing service for loading and chunking documents."""

import os
import re
from typing import List, Tuple
from pathlib import Path

import fitz  # PyMuPDF
import PyPDF2
from docx import Document

from app.utils.config import settings


class DocumentProcessor:
    """Service for processing various document types."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.supported_extensions = settings.supported_extensions
    
    def load_documents(self, documents_path: str = None) -> List[Tuple[str, str, str]]:
        """
        Load all supported documents from the specified directory.
        
        Args:
            documents_path: Path to documents directory
            
        Returns:
            List of tuples (filename, content, file_path)
        """
        if documents_path is None:
            documents_path = settings.documents_path
            
        documents = []
        documents_dir = Path(documents_path)
        
        if not documents_dir.exists():
            raise FileNotFoundError(f"Documents directory '{documents_path}' not found")
        
        for file_path in documents_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    content = self._load_document(file_path)
                    if content.strip():  # Only add non-empty documents
                        documents.append((file_path.name, content, str(file_path)))
                        print(f"Loaded document: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {str(e)}")
        
        return documents
    
    def _load_document(self, file_path: Path) -> str:
        """Load content from a single document."""
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            return self._load_txt(file_path)
        elif extension == '.pdf':
            return self._load_pdf(file_path)
        elif extension == '.docx':
            return self._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def _load_txt(self, file_path: Path) -> str:
        """Load text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load PDF file using PyMuPDF (fitz)."""
        text = ""
        try:
            # Try with PyMuPDF first
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception:
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            except Exception as e:
                raise Exception(f"Failed to read PDF {file_path}: {str(e)}")
        
        return text
    
    def _load_docx(self, file_path: Path) -> str:
        """Load DOCX file."""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def chunk_documents(self, documents: List[Tuple[str, str, str]]) -> List[dict]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of (filename, content, file_path) tuples
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        
        for filename, content, file_path in documents:
            doc_chunks = self._split_text(content)
            
            for i, chunk_text in enumerate(doc_chunks):
                chunk = {
                    'text': chunk_text,
                    'filename': filename,
                    'file_path': file_path,
                    'chunk_id': i,
                    'chunk_size': len(chunk_text)
                }
                chunks.append(chunk)
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the current chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to break at a sentence boundary
            chunk_text = text[start:end]
            
            # Look for sentence endings near the chunk boundary
            last_sentence_end = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('! '),
                chunk_text.rfind('? ')
            )
            
            if last_sentence_end > self.chunk_size * 0.5:  # If we found a good break point
                end = start + last_sentence_end + 1
                chunks.append(text[start:end].strip())
                start = end - self.chunk_overlap
            else:
                # No good sentence break found, use word boundary
                last_space = chunk_text.rfind(' ')
                if last_space > 0:
                    end = start + last_space
                
                chunks.append(text[start:end].strip())
                start = end - self.chunk_overlap
            
            # Ensure we don't go backwards
            if start < 0:
                start = 0
        
        return [chunk for chunk in chunks if chunk.strip()]