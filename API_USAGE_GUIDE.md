# 🌐 API Usage Guide - Emotion Detection Service

## 📋 API Overview

Emotion Detection API adalah REST API berbasis Flask yang dapat mendeteksi emosi dari gambar wajah menggunakan CNN model. API ini mendukung deteksi multiple faces dalam satu gambar dan mengembalikan hasil dalam format JSON.

## 🚀 Quick Start

### **Starting the API Server**

```bash
# Navigate to API directory
cd /Users/rangga/Documents/expression-detection/tracked

# Activate virtual environment
source ../venv/bin/activate

# Start the server
python3.9 emotion_api.py
```

**Server akan berjalan di:**
- **Local Access**: `http://127.0.0.1:5050`
- **Network Access**: `http://192.168.1.4:5050`
- **All Interfaces**: `http://0.0.0.0:5050`

---

## 🔗 API Endpoints

### **1. POST /detect**
**Purpose**: Deteksi emosi dari gambar yang diunggah

#### **Request Methods**

##### **Method 1: File Upload (Multipart Form)**
```bash
curl -X POST \
  -F "image=@path/to/your/image.jpg" \
  http://localhost:5050/detect
```

##### **Method 2: Base64 JSON**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD..."
  }' \
  http://localhost:5050/detect
```

##### **Method 3: Raw Base64 JSON**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "image": "/9j/4AAQSkZJRgABAQEAYABgAAD..."
  }' \
  http://localhost:5050/detect
```

#### **Request Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File/String | Yes | Image file atau base64 encoded string |

#### **Supported Image Formats**
- **File Extensions**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- **MIME Types**: `image/jpeg`, `image/png`, `image/bmp`, `image/webp`
- **Max Size**: Tidak ada batasan (akan di-resize otomatis)
- **Color Mode**: RGB atau Grayscale (akan dikonversi ke grayscale)

#### **Response Format**

##### **Success Response (200)**
```json
{
  "success": true,
  "faces_detected": 2,
  "results": [
    {
      "emotion": "happy",
      "confidence": 0.8745,
      "bounding_box": {
        "x": 120,
        "y": 80,
        "w": 150,
        "h": 150
      }
    },
    {
      "emotion": "sad",
      "confidence": 0.7234,
      "bounding_box": {
        "x": 350,
        "y": 100,
        "w": 140,
        "h": 140
      }
    }
  ]
}
```

##### **Error Response (400/500)**
```json
{
  "success": false,
  "error": "No image provided"
}
```

#### **Response Fields Explanation**

| Field | Type | Description |
|-------|------|-------------|
| `success` | Boolean | Status keberhasilan request |
| `faces_detected` | Integer | Jumlah wajah yang terdeteksi |
| `results` | Array | Array hasil deteksi per wajah |
| `emotion` | String | Label emosi: `angry`, `fear`, `happy`, `sad` |
| `confidence` | Float | Confidence score (0.0 - 1.0) |
| `bounding_box.x` | Integer | Koordinat X kiri atas face |
| `bounding_box.y` | Integer | Koordinat Y kiri atas face |
| `bounding_box.w` | Integer | Lebar bounding box |
| `bounding_box.h` | Integer | Tinggi bounding box |

---

### **2. GET /health**
**Purpose**: Health check endpoint untuk monitoring

#### **Request**
```bash
curl http://localhost:5050/health
```

#### **Response**
```json
{
  "status": "healthy",
  "message": "Emotion Detection API is running"
}
```

---

## 💻 Code Examples

### **Python Examples**

#### **Example 1: File Upload**
```python
import requests
import json

def detect_emotion_from_file(image_path, api_url="http://localhost:5050/detect"):
    """
    Deteksi emosi dari file gambar
    """
    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(api_url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print(f"Faces detected: {result['faces_detected']}")
            
            for i, face in enumerate(result['results']):
                print(f"Face {i+1}:")
                print(f"  Emotion: {face['emotion']}")
                print(f"  Confidence: {face['confidence']:.2%}")
                print(f"  Position: ({face['bounding_box']['x']}, {face['bounding_box']['y']})")
                print(f"  Size: {face['bounding_box']['w']}x{face['bounding_box']['h']}")
                
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None

# Usage
result = detect_emotion_from_file("test_image.jpg")
```

#### **Example 2: Base64 Encoding**
```python
import requests
import base64
import json

def detect_emotion_from_base64(image_path, api_url="http://localhost:5050/detect"):
    """
    Deteksi emosi menggunakan base64 encoding
    """
    try:
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare payload
        payload = {
            "image": f"data:image/jpeg;base64,{image_data}"
        }
        
        # Send request
        headers = {'Content-Type': 'application/json'}
        response = requests.post(api_url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None

# Usage
result = detect_emotion_from_base64("test_image.jpg")
```

#### **Example 3: Batch Processing**
```python
import requests
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image_batch(image_folder, api_url="http://localhost:5050/detect"):
    """
    Proses multiple gambar secara parallel
    """
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    results = {}
    
    def process_single_image(image_file):
        image_path = os.path.join(image_folder, image_file)
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(api_url, files=files)
                
            if response.status_code == 200:
                return image_file, response.json()
            else:
                return image_file, {"error": response.text}
        except Exception as e:
            return image_file, {"error": str(e)}
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_image = {executor.submit(process_single_image, img): img 
                          for img in image_files}
        
        for future in as_completed(future_to_image):
            image_file, result = future.result()
            results[image_file] = result
            print(f"Processed: {image_file}")
    
    return results

# Usage
batch_results = process_image_batch("test_images/")

# Analyze results
for image_file, result in batch_results.items():
    if 'error' not in result:
        print(f"{image_file}: {result['faces_detected']} faces detected")
    else:
        print(f"{image_file}: Error - {result['error']}")
```

### **JavaScript Examples**

#### **Example 1: Web Browser Upload**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Emotion Detection</title>
</head>
<body>
    <input type="file" id="imageInput" accept="image/*">
    <button onclick="detectEmotion()">Detect Emotion</button>
    <div id="results"></div>

    <script>
    async function detectEmotion() {
        const fileInput = document.getElementById('imageInput');
        const resultsDiv = document.getElementById('results');
        
        if (!fileInput.files[0]) {
            alert('Please select an image first');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        
        try {
            const response = await fetch('http://localhost:5050/detect', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                let html = `<h3>Faces Detected: ${result.faces_detected}</h3>`;
                
                result.results.forEach((face, index) => {
                    html += `
                        <div>
                            <h4>Face ${index + 1}</h4>
                            <p>Emotion: <strong>${face.emotion}</strong></p>
                            <p>Confidence: <strong>${(face.confidence * 100).toFixed(1)}%</strong></p>
                            <p>Position: (${face.bounding_box.x}, ${face.bounding_box.y})</p>
                            <p>Size: ${face.bounding_box.w}x${face.bounding_box.h}</p>
                        </div>
                    `;
                });
                
                resultsDiv.innerHTML = html;
            } else {
                resultsDiv.innerHTML = `<p>Error: ${result.error}</p>`;
            }
        } catch (error) {
            resultsDiv.innerHTML = `<p>Network Error: ${error.message}</p>`;
        }
    }
    </script>
</body>
</html>
```

#### **Example 2: Node.js**
```javascript
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

async function detectEmotionNodeJS(imagePath, apiUrl = 'http://localhost:5050/detect') {
    try {
        const form = new FormData();
        form.append('image', fs.createReadStream(imagePath));
        
        const response = await axios.post(apiUrl, form, {
            headers: {
                ...form.getHeaders()
            }
        });
        
        if (response.data.success) {
            console.log(`Faces detected: ${response.data.faces_detected}`);
            
            response.data.results.forEach((face, index) => {
                console.log(`Face ${index + 1}:`);
                console.log(`  Emotion: ${face.emotion}`);
                console.log(`  Confidence: ${(face.confidence * 100).toFixed(1)}%`);
                console.log(`  Bounding Box: (${face.bounding_box.x}, ${face.bounding_box.y}) ${face.bounding_box.w}x${face.bounding_box.h}`);
            });
        }
        
        return response.data;
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
        return null;
    }
}

// Usage
detectEmotionNodeJS('test_image.jpg');
```

---

## 🔧 Configuration & Customization

### **Server Configuration**

Edit `emotion_api.py` untuk mengubah konfigurasi server:

```python
# Server settings
app.run(
    host='0.0.0.0',      # Listen on all interfaces
    port=5050,           # Port number
    debug=True,          # Debug mode
    threaded=True        # Multi-threading support
)
```

### **Model Configuration**

```python
# Emotion labels (dapat diubah sesuai model)
self.emotion_labels = ['angry', 'fear', 'happy', 'sad']

# Face detection parameters
faces = self.face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,     # Increase for faster detection
    minNeighbors=5,      # Decrease for more sensitive detection
    minSize=(30, 30),    # Minimum face size
    maxSize=(300, 300)   # Maximum face size
)
```

### **CORS Configuration**

```python
# Allow specific origins
CORS(app, origins=["http://localhost:3000", "https://myapp.com"])

# Allow all origins (current setting)
CORS(app, origins="*")

# Custom CORS settings
CORS(app, 
     origins="*",
     methods=["GET", "POST"],
     allow_headers=["Content-Type", "Authorization"])
```

---

## 🐛 Troubleshooting

### **Common Issues & Solutions**

#### **1. Connection Refused**
```bash
# Error: Connection refused
# Solution: Check if server is running
ps aux | grep emotion_api

# Restart server if needed
python3.9 emotion_api.py
```

#### **2. CORS Errors**
```javascript
// Error: CORS policy blocks request
// Solution: Already handled with flask-cors
// If still occurs, check browser console for specific error
```

#### **3. 500 Internal Server Error**
```bash
# Check server logs for detailed error
tail -f /path/to/logfile

# Common causes:
# - Missing model file (model_weights.h5)
# - Missing cascade file (haarcascade_frontalface_default.xml)
# - Invalid image format
# - Memory issues
```

#### **4. No Faces Detected**
```python
# Possible solutions:
# 1. Adjust face detection parameters
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,    # More sensitive
    minNeighbors=3,      # Less strict
    minSize=(20, 20)     # Smaller minimum size
)

# 2. Check image quality
# - Ensure faces are clearly visible
# - Good lighting conditions
# - Face size > 30x30 pixels
```

#### **5. Low Confidence Scores**
```python
# Possible causes:
# - Poor image quality
# - Unusual lighting
# - Partial face occlusion
# - Expression not in training data

# Solutions:
# - Retrain model with more diverse data
# - Adjust confidence threshold
# - Improve image preprocessing
```

### **Performance Issues**

#### **Slow Response Times**
```python
# Optimization strategies:

# 1. Model optimization
# Load model once at startup (already implemented)

# 2. Image preprocessing optimization
def optimize_image_size(image, max_size=800):
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

# 3. Batch processing for multiple faces
def predict_batch(self, faces):
    if len(faces) > 1:
        batch_input = np.array(faces)
        predictions = self.model.predict(batch_input)
        return predictions
    else:
        return self.model.predict(faces[0:1])
```

#### **Memory Issues**
```python
# Memory optimization:

# 1. Clear variables after use
import gc

def detect_emotions(self, image):
    # ... detection logic ...
    
    # Clear large variables
    del image, gray, face_roi
    gc.collect()
    
    return results

# 2. Limit concurrent requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/detect', methods=['POST'])
@limiter.limit("10 per minute")
def detect():
    # ... existing code ...
```

---

## 📊 API Monitoring & Logging

### **Request Logging**

API sudah dilengkapi dengan logging. Log file akan menampilkan:

```
2025-09-XX 10:30:15 - INFO - Request: POST /detect
2025-09-XX 10:30:15 - INFO - Processing image: 1024x768 pixels
2025-09-XX 10:30:16 - INFO - Faces detected: 2
2025-09-XX 10:30:16 - INFO - Processing time: 0.847s
```

### **Health Monitoring**

```bash
# Check API health
curl http://localhost:5050/health

# Monitor with watch command
watch -n 5 'curl -s http://localhost:5050/health | jq .'

# Load testing with Apache Bench
ab -n 100 -c 10 -p test_image.json -T application/json http://localhost:5050/detect
```

### **Performance Metrics**

```python
# Add custom metrics to API
import time
from collections import defaultdict

# Global metrics storage
metrics = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'total_faces_detected': 0,
    'average_processing_time': 0,
    'emotion_distribution': defaultdict(int)
}

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)
```

---

## 🔒 Security Considerations

### **Input Validation**
```python
# File size limits
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# File type validation
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

### **Rate Limiting**
```python
# Install: pip install Flask-Limiter
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"]
)

@app.route('/detect', methods=['POST'])
@limiter.limit("30 per minute")
def detect():
    # ... existing code ...
```

### **Authentication (Optional)**
```python
# Simple API key authentication
API_KEYS = {'your-api-key-here', 'another-key'}

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key not in API_KEYS:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/detect', methods=['POST'])
@require_api_key
def detect():
    # ... existing code ...
```

---

**Last Updated**: September 2025  
**API Version**: 1.0  
**Compatibility**: Python 3.9+, Flask 2.0+