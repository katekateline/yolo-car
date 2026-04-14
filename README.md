# YOLO Car Service - Deteksi Kendaraan & Plat Nomor (Laravel Integrated)

Layanan Python (Flask) berbasis YOLOv8 untuk deteksi kendaraan, warna dominan, dan nomor plat (via PlateRecognizer API). Dirancang khusus untuk diintegrasikan dengan aplikasi Laravel.

## Fitur Utama

- **Deteksi Kendaraan (YOLOv8)**: Mendeteksi Mobil, Motor, Bus, dan Truk secara instan.
- **Warna Dominan Akurat**: Menggunakan algoritma **K-Means Clustering** yang fokus pada area body kendaraan (menghindari gangguan background/kaca).
- **Smart Plate Recognition (Async)**: 
    - Melakukan **Heuristic Crop** (hanya area plat) untuk menghemat bandwidth.
    - Pengiriman API PlateRecognizer berjalan di background (**Non-blocking**) agar respon ke Laravel super cepat.
- **Mode Scan QR**: Endpoint khusus untuk melakukan scan QR code tanpa menjalankan deteksi kendaraan.
- **Terintegrasi Laravel**: Mendukung alur `Laravel Upload -> Python Analyze -> Laravel Callback`.

## Prasyarat

- **Python 3.8+** (Disarankan 3.10 atau 3.12)
- **Koneksi Internet** (Untuk unduhan model otomatis pertama kali & PlateRecognizer API)
- **API Token PlateRecognizer** (Dapatkan di [platerecognizer.com](https://platerecognizer.com/))

## Setup & Instalasi

Ikuti langkah-langkah berikut untuk menjalankan service di lingkungan lokal:

### 1. Buat Virtual Environment (venv)

Sangat disarankan menggunakan `venv` agar library tidak bentrok dengan sistem global.

```bash
# Masuk ke folder project
cd yolo-car/yolo-service

# Buat venv (nama 'venv')
python -m venv venv

# Aktivasi venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

Pastikan `venv` sudah aktif sebelum menjalankan perintah ini:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Setup Model YOLO

Layanan ini menggunakan file model `best.pt` atau `yolov8m.pt`. 
- Jika file tidak ditemukan, Ultralytics akan **mengunduh otomatis** model standar `yolov8m.pt` saat pertama kali dijalankan.
- Jika Anda punya model custom, simpan sebagai `best.pt` di dalam folder `yolo-service/`.

### 4. Konfigurasi Environment (.env)

Buat file `.env` di dalam folder `yolo-service/` atau set environment variables berikut:

```bash
# Wajib untuk deteksi plat nomor
PLATERECOGNIZER_TOKEN=isi_token_anda_di_sini

# Opsional: Jika ingin mengirim callback otomatis ke Laravel setelah plat selesai
LARAVEL_URL=http://localhost:8000/api/detection
LARAVEL_CALLBACK_AFTER_ANALYZE=true

# Opsional: Port Flask (Default: 5000)
PORT=5000
```

## Menjalankan Service

Pastikan `venv` aktif, lalu jalankan:

```bash
python app.py
```

- **Health Check**: `GET http://localhost:5000/status`
- **Analyze Endpoint**: `POST http://localhost:5000/analyze` (Multipart: `image`, `scan_type`)

## Struktur Project

- [app.py](file:///home/kate/Documents/yolo-car/yolo-service/app.py): Entry point Flask, manajemen API, dan alur Async.
- [detector.py](file:///home/kate/Documents/yolo-car/yolo-service/detector.py): Logika deteksi YOLOv8.
- [color_detector.py](file:///home/kate/Documents/yolo-car/yolo-service/color_detector.py): Algoritma K-Means untuk warna dominan.
- [model_utils.py](file:///home/kate/Documents/yolo-car/yolo-service/model_utils.py): Utilitas pencarian path model.

## Integrasi Laravel

Untuk panduan lengkap mengenai cara menghubungkan Laravel dengan service ini (Controller, Route, Blade), silakan baca:
👉 **[LARAVEL_INTEGRATION.md](file:///home/kate/Documents/yolo-car/LARAVEL_INTEGRATION.md)**

## Troubleshooting

- **Timeout Error**: Pastikan koneksi internet stabil dan token PlateRecognizer benar. Sistem sudah dioptimalkan dengan kompresi gambar & heuristic crop.
- **Warna Tidak Akurat**: Pastikan pencahayaan cukup. Algoritma K-Means sudah difilter untuk mengabaikan bayangan gelap dan silau lampu.
- **Module Not Found**: Pastikan Anda sudah menjalankan `pip install` di dalam `venv` yang aktif.
