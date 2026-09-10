# ==========================================================
# Dockerfile for VeggieLife Computer Vision Microservice (YOLOv8)
# Optimized for Google Cloud Run (Serverless CPU Container)
# ==========================================================

FROM python:3.11-slim as base

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install minimal OS dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only wheels first to keep image size < 900MB (instead of 4GB+ GPU version)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n baseline model during build time so runtime doesn't require internet
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && mkdir -p /app/weights \
    && mv yolov8n.pt /app/weights/yolov8n.pt

# Copy source code and configuration
COPY app/ /app/app/
COPY data/ /app/data/

# Create a non-privileged user to run the application securely
RUN useradd -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run sets $PORT dynamically)
EXPOSE 8080

# Healthcheck for local and staging tests
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Start FastAPI server using uvicorn with shell execution to resolve $PORT
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
