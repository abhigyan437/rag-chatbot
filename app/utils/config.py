"""Configuration settings for the RAG chatbot system."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_title: str = "RAG Chatbot API"
    api_description: str = "A Retrieval-Augmented Generation chatbot system"
    api_version: str = "1.0.0"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7
    
    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector Store Configuration
    vector_store_path: str = "data/faiss_index"
    top_k_results: int = 3
    
    # Document Processing
    documents_path: str = "documents"
    supported_extensions: list = [".txt", ".pdf", ".docx"]
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()