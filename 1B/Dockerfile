FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Create directories
RUN mkdir -p models outputs cache logs temp

# Copy source code
COPY src/ ./src/
COPY main.py .
COPY setup.py .

# Download the model during build
RUN python -c "from setup import download_llm_model; download_llm_model()"

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main.py", "--mode", "batch"]