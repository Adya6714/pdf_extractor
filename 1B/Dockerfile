# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download models and data
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    print('Model downloaded successfully')"

RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('averaged_perceptron_tagger')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p PDFs outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

# Run the application
CMD ["python", "main.py"]
