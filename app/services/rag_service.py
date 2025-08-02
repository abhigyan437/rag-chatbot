"""RAG service for retrieval-augmented generation."""

import time
from typing import List, Dict, Any
import openai

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.models.schemas import DocumentSource
from app.utils.config import settings


class RAGService:
    """Service for Retrieval-Augmented Generation."""
    
    def __init__(self):
        """Initialize the RAG service."""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.openai_client = None
        
        # Initialize OpenAI client if API key is provided
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            print("OpenAI client initialized")
        else:
            print("Warning: OpenAI API key not provided. LLM generation will not work.")
    
    def load_vector_store(self) -> bool:
        """Load the vector store index."""
        return self.vector_store.load_index()
    
    def query(self, user_query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Process a user query using RAG.
        
        Args:
            user_query: User's question
            top_k: Number of top results to retrieve
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = settings.top_k_results
        
        try:
            # Step 1: Generate query embedding
            query_embedding = self.embedding_service.encode_text(user_query)
            
            # Step 2: Retrieve relevant documents
            search_results = self.vector_store.search(query_embedding, top_k)
            
            if not search_results:
                return {
                    "query": user_query,
                    "response": "I'm sorry, but I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            # Step 3: Prepare context and sources
            context_chunks = []
            sources = []
            
            for metadata, similarity_score in search_results:
                context_chunks.append(metadata['text'])
                
                source = DocumentSource(
                    filename=metadata['filename'],
                    chunk_id=metadata['chunk_id'],
                    similarity_score=similarity_score
                )
                sources.append(source)
            
            # Step 4: Generate response using LLM
            if self.openai_client:
                response = self._generate_llm_response(user_query, context_chunks, sources)
            else:
                response = self._generate_fallback_response(user_query, context_chunks, sources)
            
            processing_time = time.time() - start_time
            
            return {
                "query": user_query,
                "response": response,
                "sources": sources,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "query": user_query,
                "response": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "processing_time": processing_time
            }
    
    def _generate_llm_response(self, query: str, context_chunks: List[str], sources: List[DocumentSource]) -> str:
        """Generate response using OpenAI LLM."""
        
        # Prepare context
        context = "\n\n".join([f"Document {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Prepare source references
        source_refs = []
        for i, source in enumerate(sources):
            source_refs.append(f"[{i+1}] {source.filename} (chunk {source.chunk_id})")
        
        source_text = "\n".join(source_refs)
        
        # Create the prompt
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context documents. 

Context Documents:
{context}

User Question: {query}

Instructions:
1. Answer the question based solely on the information provided in the context documents
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise but comprehensive in your response
4. Reference the source documents when appropriate by mentioning the document name
5. Do not make up information not present in the context

Sources used:
{source_text}

Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.max_tokens,
                temperature=settings.temperature
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Add source references
            source_list = "\n\nSources:\n" + source_text
            
            return generated_response + source_list
            
        except Exception as e:
            return f"Error generating response with LLM: {str(e)}"
    
    def _generate_fallback_response(self, query: str, context_chunks: List[str], sources: List[DocumentSource]) -> str:
        """Generate fallback response when LLM is not available."""
        
        # Simple extraction-based response
        combined_context = " ".join(context_chunks)
        
        # Truncate if too long
        max_length = 500
        if len(combined_context) > max_length:
            combined_context = combined_context[:max_length] + "..."