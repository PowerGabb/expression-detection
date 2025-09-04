# 🔧 Technical Documentation - Emotion Detection Project

## 📋 Detailed Code Block Analysis

### 1. **CNNModel.py** - Deep Learning Architecture

#### Class Structure:
```python
class CNNModel:
    def __init__(self, input_shape=(48, 48, 1), num_classes=4)
```

#### **Method: `build_model()`**
**Purpose**: Membangun arsitektur CNN untuk klasifikasi emosi

**Architecture Details:**
```python
# Block 1: Feature Extraction
Conv2D(32, (3,3)) → BatchNorm → ReLU → MaxPool → Dropout(0.25)

# Block 2: Pattern Recognition  
Conv2D(64, (3,3)) → BatchNorm → ReLU → MaxPool → Dropout(0.25)

# Block 3: Complex Features
Conv2D(128, (3,3)) → BatchNorm → ReLU → MaxPool → Dropout(0.25)

# Block 4: High-level Features
Conv2D(256, (3,3)) → BatchNorm → ReLU → MaxPool → Dropout(0.5)

# Classification Head
Flatten → Dense(512) → Dropout(0.5) → Dense(4, softmax)
```

**Key Components:**
- **Batch Normalization**: Stabilizes training, faster convergence
- **Dropout Layers**: Prevents overfitting (0.25 → 0.5 progressive)
- **ReLU Activation**: Non-linear activation for feature learning
- **MaxPooling**: Spatial dimension reduction, translation invariance

#### **Method: `calculate_class_weights()`**
**Purpose**: Mengatasi class imbalance dalam dataset

```python
def calculate_class_weights(self, train_generator):
    # Extract all labels from generator
    labels = []
    for i in range(len(train_generator)):
        batch_x, batch_y = train_generator[i]
        labels.extend(np.argmax(batch_y, axis=1))
    
    # Calculate balanced weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))
```

**Mathematical Formula:**
```
weight_for_class_i = n_samples / (n_classes * n_samples_class_i)
```

#### **Method: `compile_model()`**
**Purpose**: Konfigurasi optimizer, loss function, dan metrics

```python
# Optimizer Configuration
Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# Loss Function
categorical_crossentropy  # Multi-class classification

# Metrics
accuracy  # Primary evaluation metric

# Callbacks
ModelCheckpoint: Save best model based on val_accuracy
ReduceLROnPlateau: Reduce LR when val_loss plateaus
```

---

### 2. **DataGenerator.py** - Data Pipeline

#### Class Structure:
```python
class DataGenerator:
    def __init__(self, data_dir, batch_size=32, target_size=(48, 48))
```

#### **Method: `create_generators()`**
**Purpose**: Membuat data generators untuk training dan validation

**Data Augmentation Strategy:**
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize to [0,1]
    rotation_range=20,        # Random rotation ±20°
    width_shift_range=0.2,    # Horizontal shift ±20%
    height_shift_range=0.2,   # Vertical shift ±20%
    shear_range=0.2,         # Shear transformation
    zoom_range=0.2,          # Random zoom ±20%
    horizontal_flip=True,     # Mirror images
    fill_mode='nearest'       # Fill strategy for transforms
)

validation_datagen = ImageDataGenerator(
    rescale=1./255           # Only normalization, no augmentation
)
```

**Generator Configuration:**
```python
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # One-hot encoding
    color_mode='grayscale',   # Single channel
    shuffle=True              # Random sampling
)
```

**Benefits of Data Augmentation:**
- **Increased Dataset Size**: 5-10x effective data multiplication
- **Improved Generalization**: Model learns invariant features
- **Reduced Overfitting**: More diverse training examples

---

### 3. **ModelEvaluator.py** - Performance Analysis

#### **Method: `plot_training_history()`**
**Purpose**: Visualisasi training progress dan detection overfitting

```python
def plot_training_history(self, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    
    # Accuracy curves  
    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
```

**Interpretation Guidelines:**
- **Good Training**: Val_loss follows train_loss closely
- **Overfitting**: Val_loss increases while train_loss decreases
- **Underfitting**: Both losses remain high and plateau

#### **Method: `plot_confusion_matrix()`**
**Purpose**: Analisis per-class performance dan bias detection

```python
def plot_confusion_matrix(self, y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Visualization with seaborn
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', 
                xticklabels=class_names, yticklabels=class_names)
```

**Key Metrics Calculated:**
- **Precision**: TP / (TP + FP) - Accuracy of positive predictions
- **Recall**: TP / (TP + FN) - Coverage of actual positives  
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Support**: Number of samples per class

---

### 4. **emotion_api.py** - Production API

#### **Class: `EmotionDetectorAPI`**

#### **Method: `__init__()`**
**Purpose**: Initialize model dan face detector

```python
def __init__(self):
    # Load trained CNN model
    self.model = tf.keras.models.load_model('model_weights.h5')
    
    # Load Haar Cascade face detector
    self.face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml'
    )
    
    # Emotion labels mapping
    self.emotion_labels = ['angry', 'fear', 'happy', 'sad']
```

#### **Method: `detect_emotions()`**
**Purpose**: Core emotion detection pipeline

```python
def detect_emotions(self, image):
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Detect faces
    faces = self.face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,    # Image pyramid scale
        minNeighbors=5,     # Minimum neighbor rectangles
        minSize=(30, 30)    # Minimum face size
    )
    
    results = []
    for (x, y, w, h) in faces:
        # Step 3: Extract and preprocess face
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=[0, -1])
        
        # Step 4: Predict emotion
        predictions = self.model.predict(face_input)
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = self.emotion_labels[emotion_idx]
        
        # Step 5: Compile results
        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'bounding_box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })
    
    return results
```

#### **Flask Routes Implementation**

#### **Route: `/detect`**
```python
@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Handle different input types
        if 'image' in request.files:
            # File upload method
            file = request.files['image']
            image_data = file.read()
        elif request.is_json and 'image' in request.json:
            # Base64 method
            base64_data = request.json['image']
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect emotions
        results = detector.detect_emotions(image)
        
        return jsonify({
            'success': True,
            'faces_detected': len(results),
            'results': results
        })
        
    except Exception as e:
        logging.error(f"Error in detect endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
```

**Input Formats Supported:**
1. **Multipart Form Data**: Direct file upload
2. **JSON Base64**: Encoded image string
3. **Data URI**: Complete data URI with MIME type

**Output Format:**
```json
{
  "success": true,
  "faces_detected": 2,
  "results": [
    {
      "emotion": "happy",
      "confidence": 0.87,
      "bounding_box": {"x": 100, "y": 50, "w": 120, "h": 120}
    },
    {
      "emotion": "sad",
      "confidence": 0.73,
      "bounding_box": {"x": 300, "y": 80, "w": 110, "h": 110}
    }
  ]
}
```

---

### 5. **__main.py** - Training Pipeline Orchestrator

#### **Main Execution Flow:**

```python
def main():
    # Step 1: Configuration
    input_shape = (48, 48, 1)
    num_classes = 4
    batch_size = 32
    target_size = (48, 48)
    
    # Step 2: Data Pipeline
    data_gen = DataGenerator(
        data_dir='archive/images',
        batch_size=batch_size,
        target_size=target_size
    )
    train_gen, val_gen = data_gen.create_generators()
    
    # Step 3: Model Architecture
    cnn_model = CNNModel(input_shape, num_classes)
    model = cnn_model.build_model()
    
    # Step 4: Class Imbalance Handling
    class_weights = cnn_model.calculate_class_weights(train_gen)
    
    # Step 5: Training Configuration
    model = cnn_model.compile_model()
    
    # Step 6: Model Training
    history = model.fit(
        train_gen,
        epochs=8,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=cnn_model.callbacks
    )
    
    # Step 7: Model Evaluation
    evaluator = ModelEvaluator()
    evaluator.plot_training_history(history.history)
    evaluator.evaluate_model(model, val_gen)
    
    # Step 8: Model Persistence
    model.save_weights('model_weights.h5')
```

---

## 🔍 Advanced Technical Concepts

### **1. Class Imbalance Handling**

**Problem**: Dataset tidak seimbang (misal: 70% happy, 10% angry, 10% fear, 10% sad)

**Solution**: Class weights adjustment
```python
# Mathematical approach
weight_i = n_total / (n_classes * n_samples_i)

# Example calculation:
# Total samples: 1000
# Happy: 700 samples → weight = 1000/(4*700) = 0.357
# Angry: 100 samples → weight = 1000/(4*100) = 2.5
# Fear: 100 samples → weight = 1000/(4*100) = 2.5  
# Sad: 100 samples → weight = 1000/(4*100) = 2.5
```

**Impact**: Model pays more attention to underrepresented classes during training.

### **2. Face Detection Pipeline**

**Haar Cascade Algorithm:**
1. **Feature Selection**: Haar-like features (edge, line, center-surround)
2. **Integral Image**: Fast feature calculation
3. **AdaBoost**: Weak classifier combination
4. **Cascade Structure**: Multi-stage rejection

**Parameters Tuning:**
```python
detectMultiScale(
    scaleFactor=1.1,     # 10% size increase per scale
    minNeighbors=5,      # Minimum detections for valid face
    minSize=(30, 30),    # Smallest detectable face
    maxSize=(300, 300)   # Largest detectable face
)
```

### **3. CNN Architecture Rationale**

**Progressive Feature Learning:**
- **Layer 1 (32 filters)**: Edges, corners, basic shapes
- **Layer 2 (64 filters)**: Textures, simple patterns
- **Layer 3 (128 filters)**: Facial components (eyes, nose, mouth)
- **Layer 4 (256 filters)**: Complex facial expressions

**Receptive Field Growth:**
```
Layer 1: 3x3 receptive field
Layer 2: 7x7 receptive field  
Layer 3: 15x15 receptive field
Layer 4: 31x31 receptive field
```

### **4. Training Optimization Strategies**

**Learning Rate Scheduling:**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # Reduce LR by 50%
    patience=3,          # Wait 3 epochs
    min_lr=1e-7         # Minimum learning rate
)
```

**Early Stopping Strategy:**
```python
ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

---

## 🚀 Performance Optimization Tips

### **1. Memory Optimization**
```python
# Reduce batch size for limited memory
batch_size = 16  # Instead of 32

# Use mixed precision training
from tensorflow.keras.mixed_precision import Policy
policy = Policy('mixed_float16')
```

### **2. Training Speed Optimization**
```python
# Use GPU if available
with tf.device('/GPU:0'):
    model.fit(...)

# Prefetch data
train_gen = train_gen.prefetch(tf.data.AUTOTUNE)
```

### **3. API Performance Optimization**
```python
# Model caching
@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model('model_weights.h5')

# Image preprocessing optimization
def preprocess_image_batch(images):
    # Batch processing for multiple faces
    return np.array([preprocess_single(img) for img in images])
```

---

## 📊 Monitoring dan Debugging

### **Training Monitoring**
```python
# Custom callback for detailed logging
class DetailedLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {logs['loss']:.4f}")
        print(f"  Train Acc: {logs['accuracy']:.4f}")
        print(f"  Val Loss: {logs['val_loss']:.4f}")
        print(f"  Val Acc: {logs['val_accuracy']:.4f}")
```

### **API Monitoring**
```python
# Request logging
@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")

# Performance monitoring
import time

@app.route('/detect', methods=['POST'])
def detect():
    start_time = time.time()
    # ... detection logic ...
    processing_time = time.time() - start_time
    logger.info(f"Processing time: {processing_time:.3f}s")
```

---

**Last Updated**: September 2025  
**Technical Level**: Advanced  
**Target Audience**: Developers, ML Engineers