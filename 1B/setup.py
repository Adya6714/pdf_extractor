#!/usr/bin/env python3
"""
Setup script to prepare the environment and download required models
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create required directories"""
    dirs = ['models', 'inputs', 'outputs', 'temp', 'cache', 'logs', 
            'src', 'src/models', 'src/processors', 'src/pipeline', 'src/utils']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True, parents=True)
        logger.info(f"Created directory: {dir_name}")

def download_llm_model():
    """Download TinyLlama model"""
    model_dir = Path("models")
    model_path = model_dir / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if model_path.exists():
        logger.info("Model already downloaded")
        return
    
    logger.info("Downloading TinyLlama model (≈670MB)...")
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    try:
        urllib.request.urlretrieve(url, model_path, reporthook=download_progress)
        logger.info(f"\nModel downloaded successfully to {model_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        sys.exit(1)

def download_progress(block_num, block_size, total_size):
    """Show download progress"""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)
    sys.stdout.write(f'\rDownload progress: {percent:.1f}%')
    sys.stdout.flush()

def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Download NLTK data
    logger.info("Downloading NLTK data...")
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True) 
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

def create_init_files():
    """Create __init__.py files"""
    init_paths = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/processors/__init__.py',
        'src/pipeline/__init__.py',
        'src/utils/__init__.py'
    ]
    
    for path in init_paths:
        Path(path).touch()
        logger.info(f"Created {path}")

def main():
    """Main setup function"""
    logger.info("Setting up PDF Processing Pipeline...")
    
    # Create directories
    create_directories()
    
    # Create __init__.py files
    create_init_files()
    
    # Install dependencies
    install_dependencies()
    
    # Download model
    download_llm_model()
    
    logger.info("\nSetup complete! You can now run the pipeline.")
    logger.info("\nTo run with Docker:")
    logger.info("  docker-compose up")
    logger.info("\nTo run directly:")
    logger.info("  python main.py")

if __name__ == "__main__":
    main()