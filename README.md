RAG Chatbot System
A comprehensive Retrieval-Augmented Generation (RAG) chatbot system built with FastAPI, FAISS, and OpenAI. This system processes documents, creates embeddings, and provides intelligent responses to user queries based on the document content.



Features

1) Document Ingestion: Supports PDF, DOCX, and TXT files
2) Vector Search: FAISS-powered similarity search for relevant content retrieval
3) LLM Integration: OpenAI GPT integration for intelligent response generation
4) RESTful API: FastAPI-based web service with automatic documentation
5) Source Attribution: Provides source references for all responses
6) Chunking Strategy: Smart text chunking with overlap for better context preservation
7) Fallback Mode: Works without OpenAI API key using simple text extraction
