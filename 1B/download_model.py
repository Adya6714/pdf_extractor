import urllib.request
import sys
from pathlib import Path

def download_with_progress(url, filename):
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min((downloaded / total_size) * 100, 100)
        sys.stdout.write(f'\rDownloading: {percent:.1f}%')
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filename, reporthook=report_progress)
    print("\nDownload complete!")

# Create models directory
Path("models").mkdir(exist_ok=True)

# Download TinyLlama
print("Downloading TinyLlama model (≈670MB)...")
model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if not Path(model_path).exists():
    download_with_progress(model_url, model_path)
else:
    print("Model already exists!")
