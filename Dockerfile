# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data outputs

# Expose port (fly.io will set PORT env var)
EXPOSE 8080

# Run the application with gunicorn (production server)
# PORT env var is set by fly.io (defaults to 8080 in fly.toml)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --threads 2 --timeout 120 --access-logfile - --error-logfile - src.dashboard.app:app"]

