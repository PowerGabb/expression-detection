

import cv2
import numpy as np
from keras.models import load_model
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import traceback
import logging

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetectorAPI:
    def __init__(self):
        # Load the trained model
        self.model = load_model('model_weights.h5')
        
        # Emotion labels
        self.emotion_labels = [
            'Angry',
            'Fear',
            'Happy',
            'Sad'
        ]
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    def detect_emotions(self, image):
        """
        Deteksi emosi dari gambar input
        
        Args:
            image: numpy array dari gambar
            
        Returns:
            dict: hasil deteksi dengan informasi wajah dan emosi
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        annotated_image = image.copy()
        
        for (x, y, w, h) in faces:
            # Extract face region
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.reshape(1, 48, 48, 1)
            roi_gray = roi_gray / 255.0
            
            # Predict emotion
            emotion_predicted = self.model.predict(roi_gray, verbose=0)
            emotion_predicted_index = np.argmax(emotion_predicted)
            emotion_label = self.emotion_labels[emotion_predicted_index]
            confidence = float(emotion_predicted[0][emotion_predicted_index])
            
            # Draw rectangle and label on image
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, f'{emotion_label} ({confidence:.2f})', 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add to results
            results.append({
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'emotion': emotion_label,
                'confidence': confidence,
                'all_predictions': {
                    label: float(emotion_predicted[0][i]) 
                    for i, label in enumerate(self.emotion_labels)
                }
            })
        
        return results, annotated_image

# Initialize detector
detector = EmotionDetectorAPI()

@app.route('/detect', methods=['POST'])
def detect_emotion():
    """
    Endpoint untuk deteksi emosi dari gambar
    
    Expected input:
    - image: base64 encoded image atau file upload
    
    Returns:
    - JSON response dengan hasil deteksi dan gambar yang sudah diannotasi
    """
    try:
        # Check if image is uploaded as file
        if 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
        # Check if image is sent as base64
        elif 'image' in request.json:
            image_data = request.json['image']
            # Remove data:image/jpeg;base64, prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert bytes to PIL Image then to OpenCV format
        pil_image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Detect emotions
        results, annotated_image = detector.detect_emotions(image)
        
        # Convert annotated image back to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            'success': True,
            'faces_detected': len(results),
            'results': results,
            'annotated_image': f'data:image/jpeg;base64,{annotated_image_b64}'
        }
        
        return jsonify(response)
        
    except Exception as e:
        # Log the full error with traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"Error in detect_emotion: {error_msg}")
        logger.error(f"Traceback: {error_traceback}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': error_traceback if app.debug else None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model is not None,
        'available_emotions': detector.emotion_labels
    })

if __name__ == '__main__':
    print("Starting Emotion Detection API...")
    print("Available endpoints:")
    print("- POST /detect - Detect emotions in image")
    print("- GET /health - Health check")
    app.run(host='0.0.0.0', port=5050, debug=True)