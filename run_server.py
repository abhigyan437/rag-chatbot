#!/usr/bin/env python3
"""
Simple script to run the RAG chatbot server.
"""

import uvicorn
from app.utils.config import settings

if __name__ == "__main__":
    print(f"Starting RAG Chatbot server on {settings.host}:{settings.port}")
    print(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    print(f"OpenAI configured: {bool(settings.openai_api_key)}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )