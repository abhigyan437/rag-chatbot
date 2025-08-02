"""Vector store service using FAISS for similarity search."""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

import faiss

from app.utils.config import settings


class VectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, store_path: str = None):
        """
        Initialize the vector store.
        
        Args:
            store_path: Path to store the FAISS index and metadata
        """
        self.store_path = Path(store_path or settings.vector_store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.metadata = []
        self.dimension = None
        
        # File paths
        self.index_path = self.store_path / "faiss_index.bin"
        self.metadata_path = self.store_path / "metadata.pkl"
    
    def create_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Create a new FAISS index with embeddings and metadata.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries for each embedding
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        self.dimension = embeddings.shape[1]
        print(f"Creating FAISS index with {len(embeddings)} embeddings, dimension: {self.dimension}")
        
        # Create FAISS index (using L2 distance, which is equivalent to cosine for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata = metadata
        
        print(f"FAISS index created with {self.index.ntotal} vectors")
    
    def save_index(self):
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            raise Exception("No index to save. Create an index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Index saved to {self.store_path}")
    
    def load_index(self) -> bool:
        """
        Load the FAISS index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.index_path.exists() or not self.metadata_path.exists():
                print("Index files not found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = self.index.d
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"Index loaded: {self.index.ntotal} vectors, dimension: {self.dimension}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (metadata, similarity_score)
        """
        if self.index is None:
            raise Exception("Index not loaded. Load or create an index first.")
        
        if top_k is None:
            top_k = settings.top_k_results
        
        # Ensure query embedding is the right shape and type
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.metadata):  # Valid index
                metadata = self.metadata[idx].copy()
                results.append((metadata, float(similarity)))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.index is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_size_mb": os.path.getsize(self.index_path) / (1024*1024) if self.index_path.exists() else 0,
            "metadata_count": len(self.metadata)
        }
    
    def rebuild_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Rebuild the entire index with new data.
        
        Args:
            embeddings: New embeddings
            metadata: New metadata
        """
        print("Rebuilding vector store index...")
        self.create_index(embeddings, metadata)
        self.save_index()
        print("Index rebuilt successfully")