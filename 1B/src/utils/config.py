# src/utils/config.py
"""
Configuration management
"""

from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class Config:
    """Global configuration"""
    # Processing parameters
    MIN_CHUNK_SIZE: int = 100
    MAX_CHUNK_SIZE: int = 1000
    MAX_SECTIONS: int = 20
    MAX_REFINED_SENTENCES: int = 5
    
    # Model paths
    MODEL_PATH: str = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Directories
    OUTPUT_DIR: str = "outputs"
    TEMP_DIR: str = "temp"
    CACHE_DIR: str = "cache"
    LOG_DIR: str = "logs"
    
    # Processing options
    USE_CACHE: bool = True
    PARALLEL_PROCESSING: bool = True
    MAX_WORKERS: int = 3
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_attr in ['OUTPUT_DIR', 'TEMP_DIR', 'CACHE_DIR', 'LOG_DIR']:
            dir_path = Path(getattr(self, dir_attr))
            dir_path.mkdir(exist_ok=True)