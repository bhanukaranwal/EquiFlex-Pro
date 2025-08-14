# Multi-stage Docker build for EquiFlex Pro
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash equiflex
WORKDIR /app
RUN chown equiflex:equiflex /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
USER equiflex
COPY --chown=equiflex:equiflex . .
CMD ["python", "-m", "src.core.engine"]

# Production stage
FROM base as production

# Copy application code
COPY --chown=equiflex:equiflex . .

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER equiflex

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "src.api.app:app"]

# API service
FROM production as api
EXPOSE 8000
CMD ["python", "-m", "src.api.app"]

# Trading engine service
FROM production as engine
CMD ["python", "-m", "src.core.engine"]