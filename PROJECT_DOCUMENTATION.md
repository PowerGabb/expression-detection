# 📊 Emotion Detection Project Documentation

## 🎯 Overview
Project ini adalah sistem deteksi emosi berbasis CNN (Convolutional Neural Network) yang dapat mengenali 4 jenis emosi dari ekspresi wajah: Angry, Fear, Happy, dan Sad. Project terdiri dari dua bagian utama: training model dan API untuk inference.

## 📁 Struktur Project

```
expression-detection/
├── tracked/                    # API dan Model Deployment
│   ├── emotion_api.py         # Flask API untuk deteksi emosi
│   ├── model_weights.h5       # Model yang sudah dilatih
│   ├── haarcascade_frontalface_default.xml  # Face detector
│   └── emotion_class/         # Folder untuk menyimpan hasil deteksi
├── tracked_train_files/       # Training dan Development
│   ├── __main.py             # Script utama untuk training
│   ├── CNNModel.py           # Definisi arsitektur CNN
│   ├── DataGenerator.py      # Data preprocessing dan augmentation
│   ├── ModelEvaluator.py     # Evaluasi dan visualisasi model
│   └── archive/images/       # Dataset training
├── requirements.txt          # Dependencies Python
├── tf-gpu.yaml              # Conda environment config
└── venv/                    # Virtual environment
```

## 🧩 Dokumentasi Blok Program

### 1. **CNNModel.py** - Arsitektur Neural Network

#### Blok Utama:
```python
class CNNModel:
    def __init__(self, input_shape, num_classes)
    def build_model(self)           # Membangun arsitektur CNN
    def compile_model(self)         # Kompilasi model dengan optimizer
    def calculate_class_weights(self) # Menghitung bobot kelas untuk mengatasi imbalance
```

**Arsitektur CNN:**
- **Input Layer**: 48x48x1 (grayscale image)
- **Conv2D Blocks**: 4 blok konvolusi dengan filter 32, 64, 128, 256
- **Batch Normalization**: Normalisasi untuk stabilitas training
- **MaxPooling**: Downsampling untuk mengurangi dimensi
- **Dropout**: Regularisasi untuk mencegah overfitting
- **Dense Layers**: Fully connected layers untuk klasifikasi
- **Output**: 4 neuron dengan softmax activation

#### Fitur Khusus:
- **Class Weights**: Mengatasi ketidakseimbangan dataset
- **Callbacks**: ModelCheckpoint dan ReduceLROnPlateau
- **Epochs**: Configurable (default: 8)

### 2. **DataGenerator.py** - Data Preprocessing

#### Blok Utama:
```python
class DataGenerator:
    def __init__(self, data_dir, batch_size, target_size)
    def create_generators(self)     # Membuat train/validation generators
    def get_class_indices(self)     # Mendapatkan mapping kelas
```

**Preprocessing Pipeline:**
- **Rescaling**: Normalisasi pixel values (0-1)
- **Data Augmentation**: Rotation, shift, zoom, flip
- **Batch Processing**: Efficient data loading
- **Class Mapping**: Automatic label encoding

### 3. **ModelEvaluator.py** - Evaluasi dan Visualisasi

#### Blok Utama:
```python
class ModelEvaluator:
    def plot_training_history(self) # Plot loss dan accuracy curves
    def plot_confusion_matrix(self) # Confusion matrix visualization
    def evaluate_model(self)        # Comprehensive model evaluation
```

**Metrics yang Dievaluasi:**
- Training/Validation Loss
- Training/Validation Accuracy
- Confusion Matrix
- Classification Report
- Per-class Precision/Recall

### 4. **emotion_api.py** - Flask API

#### Blok Utama:
```python
class EmotionDetectorAPI:
    def __init__(self)              # Load model dan face detector
    def detect_emotions(self, image) # Core detection function

# Flask Routes:
@app.route('/detect', methods=['POST'])  # Endpoint deteksi
@app.route('/health', methods=['GET'])   # Health check
```

**API Features:**
- **Face Detection**: Haar Cascade Classifier
- **Emotion Prediction**: CNN model inference
- **CORS Support**: Cross-origin requests
- **Error Handling**: Comprehensive logging
- **Multiple Input**: Base64 dan file upload

## 🚀 Panduan Pengerjaan Project dari Awal

### **Step 1: Setup Environment**

```bash
# Clone atau download project
cd expression-detection

# Buat virtual environment
python3.9 -m venv venv
source venv/bin/activate  # macOS/Linux
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip3.9 install -r requirements.txt
```

### **Step 2: Persiapan Dataset**

1. **Struktur Dataset:**
   ```
   tracked_train_files/archive/images/
   ├── 1.train/
   │   ├── angry/
   │   ├── fear/
   │   ├── happy/
   └── └── sad/
   ├── 2.validation/
   │   ├── angry/
   │   ├── fear/
   │   ├── happy/
   └── └── sad/
   └── 3.test/
       ├── angry/
       ├── fear/
       ├── happy/
       └── sad/
   ```

2. **Format Gambar:**
   - Format: JPG, PNG
   - Ukuran: Akan di-resize ke 48x48
   - Grayscale: Otomatis dikonversi

### **Step 3: Training Model**

```bash
# Masuk ke direktori training
cd tracked_train_files

# Aktifkan virtual environment
source ../venv/bin/activate

# Jalankan training
python3.9 __main.py
```

**Proses Training:**
1. **Data Loading**: DataGenerator memuat dan preprocess data
2. **Model Building**: CNNModel membangun arsitektur
3. **Class Weights**: Menghitung bobot untuk mengatasi imbalance
4. **Training**: Model dilatih dengan callbacks
5. **Evaluation**: ModelEvaluator menganalisis performa
6. **Saving**: Model terbaik disimpan sebagai `model_weights.h5`

### **Step 4: Evaluasi Model**

Setelah training selesai, akan dihasilkan:
- `conf_matrix.png`: Confusion matrix
- `plots/1.Model/`: Training curves dan metrics
- `model_weights.h5`: Model yang sudah dilatih

### **Step 5: Deploy API**

```bash
# Pindah ke direktori API
cd ../tracked

# Copy model hasil training
cp ../tracked_train_files/model_weights.h5 .

# Jalankan API
python3.9 emotion_api.py
```

**API akan berjalan di:**
- Local: `http://127.0.0.1:5050`
- Network: `http://192.168.1.4:5050`

### **Step 6: Testing API**

#### **Method 1: cURL**
```bash
# Test dengan file upload
curl -X POST -F "image=@test_image.jpg" http://localhost:5050/detect

# Health check
curl http://localhost:5050/health
```

#### **Method 2: Python Script**
```python
import requests
import base64

# Upload file
with open('test_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5050/detect', files=files)
    print(response.json())

# Base64 method
with open('test_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    payload = {'image': f'data:image/jpeg;base64,{image_data}'}
    response = requests.post('http://localhost:5050/detect', json=payload)
    print(response.json())
```

## 🔧 Konfigurasi dan Customization

### **Mengubah Jumlah Epochs**
Edit `CNNModel.py`, line ~160:
```python
self.model.fit(
    train_generator,
    epochs=8,  # Ubah nilai ini
    ...
)
```

### **Menambah Kelas Emosi Baru**
1. Tambah folder di dataset dengan nama kelas baru
2. Update `emotion_labels` di `emotion_api.py`
3. Retrain model

### **Mengubah Input Size**
Edit `__main.py`:
```python
input_shape = (64, 64, 1)  # Ubah dari (48, 48, 1)
target_size = (64, 64)     # Ubah dari (48, 48)
```

## 🐛 Troubleshooting

### **Common Issues:**

1. **ModuleNotFoundError**
   ```bash
   # Pastikan virtual environment aktif
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **CORS Error**
   - Sudah ditangani dengan flask-cors
   - API dapat diakses dari browser mana pun

3. **Model Bias (selalu prediksi 'happy')**
   - Sudah ditangani dengan class weights
   - Retrain model jika masih terjadi

4. **Memory Error**
   ```python
   # Kurangi batch_size di DataGenerator
   batch_size = 16  # Dari 32
   ```

5. **Low Accuracy**
   - Tambah epochs
   - Cek kualitas dataset
   - Adjust learning rate

## 📊 Expected Results

### **Training Metrics:**
- **Validation Accuracy**: ~65-70%
- **Training Time**: 10-30 menit (tergantung hardware)
- **Model Size**: ~80MB

### **API Performance:**
- **Response Time**: <2 detik per gambar
- **Supported Formats**: JPG, PNG, WebP
- **Max Image Size**: Tidak ada batasan (akan di-resize)

## 🔄 Workflow Development

1. **Data Collection** → Kumpulkan dataset emosi
2. **Data Preprocessing** → Organize dan clean data
3. **Model Training** → Train CNN dengan class weights
4. **Model Evaluation** → Analyze performance metrics
5. **API Development** → Deploy model sebagai REST API
6. **Testing** → Test API dengan berbagai input
7. **Production** → Deploy ke server production

## 📝 Best Practices

1. **Dataset Quality**: Pastikan gambar berkualitas baik dan balanced
2. **Validation**: Selalu gunakan validation set terpisah
3. **Monitoring**: Monitor training curves untuk detect overfitting
4. **Versioning**: Simpan model dengan timestamp/version
5. **Documentation**: Update dokumentasi setiap ada perubahan
6. **Testing**: Test API secara menyeluruh sebelum production

---

**Author**: AI Assistant  
**Last Updated**: September 2025  
**Version**: 1.0