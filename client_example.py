#!/usr/bin/env python3
"""
Example client script to interact with the RAG chatbot API.
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def check_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ API is running")
            return True
        else:
            print("✗ API health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure the server is running.")
        return False

def get_stats():
    """Get vector store statistics."""
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("\n📊 System Statistics:")
            print(f"Vector store status: {stats['vector_store']['status']}")
            if stats['vector_store']['status'] == 'loaded':
                print(f"Total vectors: {stats['vector_store']['total_vectors']}")
                print(f"Embedding dimension: {stats['vector_store']['dimension']}")
                print(f"Index size: {stats['vector_store']['index_size_mb']:.2f} MB")
            print(f"OpenAI configured: {stats['settings']['openai_configured']}")
            print(f"Embedding model: {stats['settings']['embedding_model']}")
            return stats
        else:
            print("✗ Failed to get statistics")
            return None
    except Exception as e:
        print(f"✗ Error getting statistics: {e}")
        return None

def ingest_documents():
    """Trigger document ingestion."""
    print("\n📚 Starting document ingestion...")
    try:
        response = requests.post(f"{BASE_URL}/ingest")
        if response.status_code == 200:
            result = response.json()
            print("✓ Document ingestion completed!")
            print(f"Documents processed: {result['documents_processed']}")
            print(f"Chunks created: {result['chunks_created']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            return True
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            print(f"✗ Ingestion failed: {error_detail}")
            return False
    except Exception as e:
        print(f"✗ Error during ingestion: {e}")
        return False

def query_system(query):
    """Send a query to the RAG system."""
    try:
        payload = {"query": query}
        response = requests.post(f"{BASE_URL}/query", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n❓ Query: {result['query']}")
            print(f"🤖 Response: {result['response']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
            
            if result['sources']:
                print("\n📖 Sources:")
                for source in result['sources']:
                    print(f"  - {source['filename']} (chunk {source['chunk_id']}, "
                          f"similarity: {source['similarity_score']:.3f})")
            
            return result
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            print(f"✗ Query failed: {error_detail}")
            return None
            
    except Exception as e:
        print(f"✗ Error during query: {e}")
        return None

def interactive_mode():
    """Interactive chat mode."""
    print("\n🚀 Interactive mode started. Type 'quit' to exit.")
    print("Enter your questions about the documents:")
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
                
            result = query_system(query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    """Main function."""
    print("🤖 RAG Chatbot Client")
    print("=" * 30)
    
    # Check if API is running
    if not check_health():
        return
    
    # Get current statistics
    stats = get_stats()
    
    # Check if vector store is loaded
    if not stats or stats['vector_store']['status'] != 'loaded':
        print("\n⚠️  Vector store not loaded. Running document ingestion...")
        if not ingest_documents():
            print("❌ Cannot proceed without successful ingestion.")
            return
    
    # Example queries
    example_queries = [
        "What is machine learning?",
        "Explain deep learning and neural networks",
        "What are the main types of machine learning?",
        "How is Python used in data science?",
        "What is artificial intelligence?"
    ]
    
    print("\n🔍 Running example queries:")
    for query in example_queries[:2]:  # Run first 2 examples
        query_system(query)
        time.sleep(1)  # Small delay between queries
    
    # Start interactive mode
    interactive_mode()

if __name__ == "__main__":
    main()