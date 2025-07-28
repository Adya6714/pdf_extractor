# Use slim Python base for smaller image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (leverages layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy rest of your code
COPY . .

# Download the TinyLlama model during build
# (skip if you prefer to mount it at runtime)
RUN mkdir -p models \
 && curl -L -o models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
      https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
 && head -c 4 models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | grep GGUF

# Default command: run batch on all Collections
CMD ["python3", "main.py", "--mode", "batch", "--collections-dir", "Collections"]