# src/processors/embedding_processor.py
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm

class EmbeddingProcessor:
    """Handles all embedding operations efficiently"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Load model
        self.model = SentenceTransformer(model_name)
        
        # Check model size
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e9
        print(f"Model size: {model_size:.2f} GB")
        
        # Set to CPU mode
        self.device = 'cpu'
        self.model.to(self.device)
        
        # Initialize FAISS index
        self.index = None
        self.indexed_chunks = []
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, 
                    show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings efficiently"""
        embeddings = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=False,
                device=self.device,
                normalize_embeddings=True
            )
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def build_index(self, embeddings: np.ndarray, chunks: List):
        """Build FAISS index for fast similarity search"""
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (equivalent to cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        self.indexed_chunks = chunks
        
        print(f"Built index with {self.index.ntotal} vectors")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Search for similar chunks using FAISS"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Ensure query embedding is normalized
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), k
        )
        
        # Return results as (index, similarity_score) tuples
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], distances[0])]
        return results
    
    def calculate_pairwise_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarity matrix"""
        return cosine_similarity(embeddings)
    
    def create_query_embedding(self, query: str, persona_keywords: List[str]) -> np.ndarray:
        """Create enhanced query embedding with persona context"""
        # Combine query with persona keywords
        enhanced_query = f"{query} {' '.join(persona_keywords[:5])}"
        
        # Encode
        embedding = self.model.encode(
            enhanced_query,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        return embedding