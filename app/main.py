"""FastAPI application for RAG chatbot."""

import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    QueryRequest, 
    QueryResponse, 
    HealthResponse, 
    DocumentIngestionResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.rag_service import RAGService
from app.utils.config import settings


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
rag_service = RAGService()
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("Starting RAG Chatbot API...")
    
    # Try to load existing vector store
    if rag_service.load_vector_store():
        print("Vector store loaded successfully")
    else:
        print("No existing vector store found. Use /ingest endpoint to create one.")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="healthy",
        message="RAG Chatbot API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="RAG Chatbot API is running"
    )


@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    stats = vector_store.get_stats()
    return {
        "vector_store": stats,
        "settings": {
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k_results": settings.top_k_results,
            "openai_configured": bool(settings.openai_api_key)
        }
    }


@app.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_documents(background_tasks: BackgroundTasks):
    """
    Ingest documents and create/update the vector store.
    This is a background task as it can take some time.
    """
    try:
        start_time = time.time()
        
        # Load documents
        print("Loading documents...")
        documents = document_processor.load_documents()
        
        if not documents:
            raise HTTPException(
                status_code=404, 
                detail=f"No supported documents found in '{settings.documents_path}' directory"
            )
        
        # Chunk documents
        print("Chunking documents...")
        chunks = document_processor.chunk_documents(documents)
        
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail="No valid chunks created from documents"
            )
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_service.encode_texts(texts)
        
        # Create/update vector store
        print("Creating vector store...")
        vector_store.rebuild_index(embeddings, chunks)
        
        processing_time = time.time() - start_time
        
        # Refresh the RAG service vector store
        rag_service.load_vector_store()
        
        return DocumentIngestionResponse(
            status="success",
            documents_processed=len(documents),
            chunks_created=len(chunks),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    try:
        # Check if vector store is available
        if not rag_service.load_vector_store():
            raise HTTPException(
                status_code=503, 
                detail="Vector store not available. Please run document ingestion first using /ingest endpoint."
            )
        
        # Process the query
        result = rag_service.query(request.query)
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Alternative endpoint for chat functionality (same as query)."""
    return await query_documents(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )