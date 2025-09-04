# Emotion Detection API

API Flask untuk deteksi emosi dari gambar menggunakan model CNN yang sudah dilatih.

## Fitur

- ✅ Deteksi wajah menggunakan Haar Cascade Classifier
- ✅ Prediksi emosi menggunakan model CNN terlatih
- ✅ Mengembalikan gambar dengan kotak deteksi dan label emosi
- ✅ Support input base64 dan file upload
- ✅ Confidence score untuk setiap prediksi
- ✅ Semua probabilitas emosi untuk analisis lebih detail

## Instalasi

1. **Install dependencies baru:**
   ```bash
   pip install Flask==2.3.3
   ```
   
   Atau install semua dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Pastikan file model dan classifier ada:**
   - `model_weights.h5` (model yang sudah dilatih)
   - `haarcascade_frontalface_default.xml` (face classifier)

## Menjalankan API

```bash
cd tracked
python emotion_api.py
```

API akan berjalan di `http://localhost:5000`

## Endpoints

### 1. POST `/detect`

Deteksi emosi dari gambar.

**Input Options:**

**Option A: File Upload**
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/detect
```

**Option B: Base64 JSON**
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"image": "data:image/jpeg;base64,YOUR_BASE64_STRING"}' \
     http://localhost:5000/detect
```

**Response:**
```json
{
  "success": true,
  "faces_detected": 1,
  "results": [
    {
      "bbox": {
        "x": 100,
        "y": 50,
        "width": 150,
        "height": 150
      },
      "emotion": "Happy",
      "confidence": 0.85,
      "all_predictions": {
        "Angry": 0.02,
        "Disgust": 0.01,
        "Fear": 0.03,
        "Happy": 0.85,
        "Neutral": 0.05,
        "Sad": 0.02,
        "Surprise": 0.02
      }
    }
  ],
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

### 2. GET `/health`

Health check endpoint.

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "available_emotions": [
    "Angry", "Fear", "Happy", "Sad"
  ]
}
```

## Testing

Gunakan script test yang disediakan:

```bash
python test_api.py
```

Script ini akan:
1. Check health API
2. Test dengan file gambar (base64 dan file upload)
3. Menyimpan hasil annotated image

## Contoh Penggunaan Python

```python
import requests
import base64

# Method 1: File upload
with open('test_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/detect', files=files)
    result = response.json()

# Method 2: Base64
with open('test_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    payload = {'image': f'data:image/jpeg;base64,{image_data}'}
    response = requests.post('http://localhost:5000/detect', json=payload)
    result = response.json()

print(f"Detected {result['faces_detected']} faces")
for face in result['results']:
    print(f"Emotion: {face['emotion']} (confidence: {face['confidence']:.3f})")
```

## Response Fields

- `success`: Boolean indicating if request was successful
- `faces_detected`: Number of faces found in image
- `results`: Array of detection results for each face
  - `bbox`: Bounding box coordinates (x, y, width, height)
  - `emotion`: Predicted emotion label
  - `confidence`: Confidence score (0-1) for the predicted emotion
  - `all_predictions`: Probability scores for all emotion classes
- `annotated_image`: Base64 encoded image with drawn bounding boxes and labels

## Supported Emotions

1. **Angry** - Marah
2. **Fear** - Takut
3. **Happy** - Senang
4. **Sad** - Sedih

## Error Handling

API akan mengembalikan error response jika:
- Tidak ada gambar yang dikirim
- Format gambar tidak valid
- Model tidak dapat memproses gambar
- Server error

**Error Response:**
```json
{
  "success": false,
  "error": "Error message here"
}
```

## Tips Penggunaan

1. **Format Gambar**: Mendukung JPG, PNG, dan format umum lainnya
2. **Ukuran Gambar**: Tidak ada batasan khusus, tapi gambar besar akan memakan waktu lebih lama
3. **Kualitas Deteksi**: Hasil terbaik dengan:
   - Wajah yang jelas dan tidak terlalu kecil
   - Pencahayaan yang baik
   - Wajah menghadap ke depan
4. **Performance**: API dapat memproses multiple faces dalam satu gambar

## Troubleshooting

**API tidak bisa start:**
- Pastikan semua dependencies terinstall
- Check apakah file `model_weights.h5` dan `haarcascade_frontalface_default.xml` ada
- Pastikan port 5000 tidak digunakan aplikasi lain

**Deteksi tidak akurat:**
- Coba dengan gambar yang lebih jelas
- Pastikan wajah cukup besar dalam gambar
- Check pencahayaan gambar

**Memory error:**
- Resize gambar ke ukuran lebih kecil sebelum dikirim
- Restart API jika sudah memproses banyak gambar