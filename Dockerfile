# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project first
COPY . .

# Create and activate virtual environment, then install dependencies
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Check if model exists, if not show error message
RUN if [ ! -f "/app/tracked/model_weights.h5" ]; then \
        echo "ERROR: model_weights.h5 not found in /app/tracked/"; \
        echo "Please ensure model_weights.h5 is available by:"; \
        echo "1. Copying it to tracked/ directory before building"; \
        echo "2. Mounting it as volume: -v /path/to/model_weights.h5:/app/tracked/model_weights.h5"; \
        echo "3. Using docker-compose with volume configuration"; \
        exit 1; \
    fi

# Expose port
EXPOSE 5050

# Set environment variables
ENV FLASK_APP=tracked/emotion_api.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5050
ENV PYTHONPATH=/app

# Activate virtual environment and run the application
CMD ["/bin/bash", "-c", "source venv/bin/activate && cd /app/tracked && python emotion_api.py"]