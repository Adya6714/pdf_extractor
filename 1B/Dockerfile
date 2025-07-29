# Multi-stage build to reduce final image size
FROM --platform=linux/amd64 python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Final stage - minimal runtime image
FROM --platform=linux/amd64 python:3.9-slim

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Create necessary directories
RUN mkdir -p /app/Collections /app/outputs /app/logs /app/models

# Copy the model file (baked into image for offline use)
COPY models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf /app/models/

# Copy source code
COPY src/ /app/src/
COPY *.py /app/

# Copy collections (this bakes the data into the image)
COPY Collections/ /app/Collections/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# Default command to run the pipeline
CMD ["python", "main.py"] 
