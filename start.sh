#!/bin/bash

# Activate virtual environment
source /app/venv/bin/activate

echo "Starting Emotion Detection API..."
echo "Checking for required files..."

# Change to tracked directory
cd /app/tracked

# Check if model file exists
if [ ! -f "model_weights.h5" ]; then
    echo "Error: Model file 'model_weights.h5' not found in /app/tracked/"
    echo "Please ensure the model file is available before running the container."
    echo "You can:"
    echo "1. Copy model_weights.h5 to the tracked/ directory before building"
    echo "2. Mount the model file as a volume when running the container"
    exit 1
else
    echo "Model file found: model_weights.h5"
fi

# Check for cascade file
cd /app/tracked
if [ ! -f "haarcascade_frontalface_default.xml" ]; then
    echo "Error: haarcascade_frontalface_default.xml not found"
    exit 1
else
    echo "Cascade file found: haarcascade_frontalface_default.xml"
fi

echo "All required files are present."
echo "Starting Flask application on port 5050..."
echo "API will be available at http://0.0.0.0:5050"
echo "Health check endpoint: http://0.0.0.0:5050/health"
echo "Detection endpoint: http://0.0.0.0:5050/detect"

# Start the Flask application
python emotion_api.py