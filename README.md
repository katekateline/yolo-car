# YOLO Car Detection – IP Webcam → Laravel

Deteksi kendaraan (mobil, motor, truk, bus) dari **IP Webcam** dengan YOLO, deteksi **warna dominan**, lalu kirim data ke **Laravel** lewat `POST /api/detection`. Model yang dipakai: **yolov8m.pt** (di folder root project).

## Fitur

- Koneksi ke IP Webcam (`http://IP_HP:8080/video`) dengan reconnect otomatis
- Deteksi kendaraan: **car**, **motorcycle**, **truck**, **bus** (YOLO)
- Deteksi warna dominan: merah, biru, hitam, putih, abu, silver, hijau, kuning, orange, ungu, coklat
- Cooldown 3 detik agar tidak spam ke Laravel
- POST otomatis ke Laravel `POST /api/detection` (JSON)
- Health check: `GET /status` (Flask); deteksi berjalan di background thread
- Model: **yolov8m.pt** di folder root project (prioritas pertama)

## Struktur Project

```
yolo-car/
├── yolov8m.pt          ← Model YOLO (taruh di sini)
├── README.md
├── LARAVEL_INTEGRATION.md
├── .gitignore
└── yolo_service/
    ├── app.py           # Entry: Flask /status + background detection
    ├── detector.py      # YOLO + filter kendaraan
    ├── color_detector.py# Warna dominan dari bbox
    └── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
cd yolo_service
pip install -r requirements.txt
```

### 2. Model YOLO (yolov8m.pt)

Letakkan file **yolov8m.pt** di **folder root project** (sejajar dengan folder `yolo_service/`):

```
yolo-car/
├── yolov8m.pt    ← di sini
└── yolo_service/
```

Atau download:

```bash
# Dari folder root project (yolo-car)
# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt" -OutFile "yolov8m.pt"
```

Atau manual: [Ultralytics assets](https://github.com/ultralytics/assets/releases) → download **yolov8m.pt** → simpan di folder **yolo-car** (root).

### 3. Konfigurasi

Edit `yolo_service/app.py` atau set environment variable:

- **WEBCAM_URL** – URL stream IP Webcam (ganti dengan IP HP Anda)
- **LARAVEL_URL** – Endpoint Laravel (default: `http://localhost:8000/api/detection`)

Contoh di `app.py`:

```python
WEBCAM_URL = "http://192.168.1.100:8080/video"  # Ganti dengan IP HP
LARAVEL_URL = "http://localhost:8000/api/detection"
```

Atau lewat env:

```bash
# Windows PowerShell
$env:WEBCAM_URL="http://192.168.1.5:8080/video"
$env:LARAVEL_URL="http://localhost:8000/api/detection"
```

### 4. Jalankan service

```bash
cd yolo_service
python app.py
```

- Service: `http://localhost:5000`
- Health: `GET http://localhost:5000/status` → `{"status":"running"}`
- Deteksi berjalan otomatis di background (baca webcam → YOLO → POST ke Laravel)

## Endpoint Python (Flask)

| Method | Path    | Keterangan                |
|--------|--------|----------------------------|
| GET    | /status | Health check, return `{"status":"running"}` |

Tidak ada endpoint untuk upload gambar. Semua deteksi dari stream IP Webcam dan dikirim ke Laravel.

## Format data ke Laravel

Python mengirim **POST** ke Laravel dengan body JSON:

```json
{
  "vehicle_type": "car",
  "color": "hitam",
  "confidence": 0.87,
  "timestamp": "2026-02-22 14:00:00"
}
```

Laravel harus menyediakan **POST /api/detection** untuk menerima data ini. Panduan dan **prompt untuk update Laravel** ada di **LARAVEL_INTEGRATION.md**.

## Integrasi Laravel

Lihat **LARAVEL_INTEGRATION.md** untuk:

- Kontrak API (POST /api/detection)
- **Prompt siap pakai** untuk mengupdate Laravel (receive JSON, validasi, simpan/auto-fill form)
- Contoh route, controller, dan auto-fill form

## Troubleshooting

### Model tidak ditemukan

- Pastikan **yolov8m.pt** ada di folder **root project** (`yolo-car/yolov8m.pt`), bukan di dalam `yolo_service/`.
- Atau taruh **best.pt** di `yolo_service/` sebagai alternatif.

### Webcam tidak konek

- Cek IP HP dan port (biasanya 8080).
- Pastikan HP dan PC dalam jaringan yang sama.
- Aplikasi IP Webcam di HP harus aktif dan stream video menyala.

### Data tidak sampai ke Laravel

- Pastikan Laravel jalan di `http://localhost:8000` (atau URL yang di-set di `LARAVEL_URL`).
- Pastikan route **POST /api/detection** ada dan return 2xx.
- Cek log console Python: `[INFO] Data dikirim ke Laravel` vs `[ERROR] Gagal kirim data`.

### Port 5000 sudah dipakai

Edit `yolo_service/app.py` baris terakhir:

```python
app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)  # ganti port
```
