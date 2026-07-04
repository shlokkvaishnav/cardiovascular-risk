# Use slim python image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    GIT_PYTHON_REFRESH=quiet

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for cache optimization
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src
COPY scripts/ ./scripts
COPY models/ ./models
COPY config/ ./config
COPY data/ ./data
COPY logs/ ./logs

# Bake a trained model + SHAP background into the image at build time, so the
# container is immediately servable without depending on a runtime volume mount.
RUN python scripts/train.py --config config/config.yaml

# Create a non-root user, own the artifacts it just produced
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
