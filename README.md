RAG Chatbot System
A comprehensive Retrieval-Augmented Generation (RAG) chatbot system built with FastAPI, FAISS, and OpenAI. This system processes documents, creates embeddings, and provides intelligent responses to user queries based on the document content.
🚀 Features

Document Ingestion: Supports PDF, DOCX, and TXT files
Vector Search: FAISS-powered similarity search for relevant content retrieval
LLM Integration: OpenAI GPT integration for intelligent response generation
RESTful API: FastAPI-based web service with automatic documentation
Source Attribution: Provides source references for all responses
Chunking Strategy: Smart text chunking with overlap for better context preservation
Fallback Mode: Works without OpenAI API key using simple text extraction