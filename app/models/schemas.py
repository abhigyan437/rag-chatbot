"""Pydantic models for request/response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?"
            }
        }


class DocumentSource(BaseModel):
    """Model for document source information."""
    filename: str = Field(..., description="Source document filename")
    chunk_id: int = Field(..., description="Chunk identifier")
    similarity_score: float = Field(..., description="Similarity score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "ml_basics.pdf",
                "chunk_id": 1,
                "similarity_score": 0.85
            }
        }


class QueryResponse(BaseModel):
    """Response model for chat queries."""
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Generated response")
    sources: List[DocumentSource] = Field(..., description="Source documents used")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence...",
                "sources": [
                    {
                        "filename": "ml_basics.pdf",
                        "chunk_id": 1,
                        "similarity_score": 0.85
                    }
                ],
                "processing_time": 1.23
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "RAG Chatbot API is running"
            }
        }


class DocumentIngestionResponse(BaseModel):
    """Response model for document ingestion."""
    status: str = Field(..., description="Ingestion status")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "documents_processed": 5,
                "chunks_created": 150,
                "processing_time": 45.67
            }
        }