# Car Detection API dengan YOLO

API Python untuk deteksi mobil menggunakan YOLO yang terintegrasi dengan Laravel.

## Fitur

- ✅ Deteksi mobil dari gambar yang diupload
- ✅ Tracking mobil terbaru (auto cleanup mobil lama >24 jam)
- ✅ Deteksi warna dan tipe kendaraan
- ✅ RESTful API dengan Flask
- ✅ CORS enabled untuk integrasi Laravel

## Setup

### 1. Install Dependencies

```bash
# Pastikan setuptools terinstall terlebih dahulu
python -m pip install --upgrade pip setuptools wheel

# Install semua dependencies
pip install -r requirements.txt
```

### 2. Download Model YOLO

Pastikan model YOLO ada di folder `model/`. Anda bisa download dari:

```bash
# Buat folder model jika belum ada
mkdir model

# Download model (pilih salah satu)
# Model kecil (cepat, akurasi standar)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P model/

# Model sedang (recommended)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -P model/
```

Atau download manual dari: https://github.com/ultralytics/assets/releases

### 3. Jalankan API

```bash
python api.py
```

API akan berjalan di `http://localhost:5000`

## Endpoints

### POST /detect
Deteksi mobil dari gambar

**Request:**
```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@path/to/image.jpg" \
  -F "image_path=public/images/image.jpg"
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-02-13T10:30:00",
  "total_detections": 3,
  "cars_detected": 2,
  "latest_cars": [...],
  "all_detections": [...]
}
```

### GET /latest-cars
Mendapatkan daftar mobil terbaru

**Response:**
```json
{
  "success": true,
  "total_cars": 5,
  "removed_old_cars": 2,
  "cars": [...]
}
```

### GET /health
Health check endpoint

## Integrasi dengan Laravel

Lihat file `LARAVEL_INTEGRATION.md` untuk panduan lengkap integrasi dengan Laravel.

## Troubleshooting

### Error: Cannot import 'setuptools.build_meta'
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Error: Model not found
Pastikan file model YOLO ada di folder `model/` dengan nama seperti:
- `model/yolov8n.pt`
- `model/yolov8m.pt`
- `model/yolov8s.pt`

### Port sudah digunakan
Edit file `api.py` dan ubah port di bagian akhir:
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Ubah port ke 5001
```
