# config.py
import os

class Config:
    # Model settings
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    MAX_MODEL_SIZE_GB = 1.0
    
    # Processing constraints
    MAX_PROCESSING_TIME = 60  # seconds
    CPU_ONLY = True
    
    # Document processing
    MIN_CHUNK_SIZE = 50
    MAX_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Scoring weights
    SEMANTIC_WEIGHT = 0.4
    KEYWORD_WEIGHT = 0.3
    STRUCTURAL_WEIGHT = 0.15
    CONTEXTUAL_WEIGHT = 0.15
    
    # Output settings
    MAX_SECTIONS = 15
    MAX_REFINED_SENTENCES = 3
    
    # Paths
    PDF_DIR = "PDFs"
    OUTPUT_DIR = "outputs"
