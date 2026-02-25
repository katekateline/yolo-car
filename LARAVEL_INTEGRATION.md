# Integrasi Laravel dengan YOLO Service (IP Webcam → Python → Laravel)

Arsitektur: **Python** mengambil foto dari **IP Webcam** tiap beberapa detik, menyimpan ke folder **storage**, mendeteksi kendaraan (YOLO + warna). Foto **tanpa kendaraan** bisa dihapus otomatis (opsional). Data deteksi bisa dikirim ke Laravel lewat **POST /api/detection** atau dibaca lewat endpoint Python.

---

## 1. Endpoint yang dikeluarkan oleh API Python (YOLO Service)

API ini berjalan di **http://localhost:5000** (bisa diubah). Laravel atau frontend **memanggil** endpoint berikut ke Python.

### 1.1 GET /status

**Deskripsi:** Health check untuk mengecek apakah service sedang jalan.

| Item    | Nilai |
|--------|--------|
| Method  | `GET` |
| URL     | `http://localhost:5000/status` |
| Headers | (optional) |

**Response (200 OK):**

```json
{
  "status": "running"
}
```

**Contoh pemanggilan (Laravel PHP):**

```php
$response = Http::get('http://localhost:5000/status');
// $response->json() => ['status' => 'running']
```

---

### 1.2 GET /latest

**Deskripsi:** Mengambil hasil analisis deteksi terbaru (foto terakhir yang ada kendaraan + daftar kendaraan beserta tipe dan warna).

| Item    | Nilai |
|--------|--------|
| Method  | `GET` |
| URL     | `http://localhost:5000/latest` |
| Headers | (optional) |

**Response (200 OK) – ketika sudah pernah ada deteksi:**

```json
{
  "photo": "storage/20260223_153000.jpg",
  "timestamp": "2026-02-23 15:30:00",
  "updated_at": "2026-02-23T15:30:00.123456",
  "detections": [
    {
      "vehicle_type": "car",
      "color": "orange",
      "confidence": 0.87,
      "bbox": [100, 200, 400, 500]
    }
  ]
}
```

**Response (200 OK) – ketika belum pernah ada deteksi:**

```json
{
  "photo": null,
  "timestamp": null,
  "updated_at": null,
  "detections": [],
  "message": "Belum ada deteksi"
}
```

**Format field response:**

| Field         | Tipe   | Keterangan |
|---------------|--------|------------|
| `photo`       | string | Path relatif ke foto di storage (contoh: `storage/20260223_153000.jpg`) |
| `timestamp`   | string | Waktu deteksi, format `YYYY-MM-DD HH:MM:SS` |
| `updated_at`  | string | Waktu update terakhir (ISO 8601) |
| `detections`  | array  | Daftar kendaraan terdeteksi |
| `detections[].vehicle_type` | string | `car`, `motorcycle`, `truck`, `bus` |
| `detections[].color`        | string | Warna: `merah`, `biru`, `hitam`, `putih`, `abu`, `silver`, `hijau`, `kuning`, `orange`, `ungu`, `coklat` |
| `detections[].confidence`   | float  | Confidence YOLO 0–1 |
| `detections[].bbox`        | array  | `[x1, y1, x2, y2]` bounding box |

**Contoh pemanggilan (Laravel PHP):**

```php
$response = Http::get('http://localhost:5000/latest');
$data = $response->json();
// $data['detections'], $data['photo'], $data['timestamp'], dll.
```

---

## 2. Ringkasan endpoint API Python

| Method | Endpoint  | Deskripsi                    |
|--------|-----------|------------------------------|
| GET    | /status   | Cek service hidup            |
| GET    | /latest   | Hasil analisis deteksi terbaru (foto + detections) |

---

## 3. Setup Python (YOLO Service)

### 3.1 Install dependencies

```bash
cd yolo_service
pip install -r requirements.txt
```

### 3.2 Model YOLO

Letakkan **yolov8m.pt** di folder root project, atau **best.pt** di dalam `yolo_service/`. Prioritas: yolov8m.pt (root) → best.pt (yolo_service) → default.

### 3.3 Konfigurasi (app.py / env)

- **WEBCAM_URL** – URL stream IP Webcam (contoh: `http://192.168.1.100:8080/video`)
- **LARAVEL_URL** – Endpoint Laravel untuk terima deteksi: `http://localhost:8000/api/detection`
- **CAPTURE_INTERVAL_SEC** – Interval pengambilan foto (detik), default 5
- **DELETE_IMAGE_IF_NO_VEHICLE** – `True` = hapus foto jika tidak ada kendaraan; `False` = simpan semua foto

### 3.4 Jalankan

```bash
cd yolo_service
python app.py
```

Service: `http://localhost:5000`. Deteksi berjalan otomatis (foto tiap N detik → simpan ke storage → deteksi → jika tidak ada kendaraan dan opsi aktif, hapus foto).

---

## 4. Kontrak API Laravel (yang harus disediakan Laravel)

Jika mode Laravel aktif, Python akan **memanggil** endpoint ini saat ada kendaraan terdeteksi (dengan cooldown):

**POST** `http://localhost:8000/api/detection`

**Headers:** `Content-Type: application/json`

**Body (JSON):**

```json
{
  "vehicle_type": "car",
  "color": "hitam",
  "confidence": 0.87,
  "timestamp": "2026-02-22 14:00:00"
}
```

| Field          | Tipe   | Keterangan |
|----------------|--------|------------|
| `vehicle_type` | string | `car`, `motorcycle`, `truck`, `bus` |
| `color`        | string | Warna (merah, biru, hitam, putih, abu, silver, hijau, kuning, orange, ungu, coklat) |
| `confidence`   | float  | 0–1 |
| `timestamp`    | string | `YYYY-MM-DD HH:MM:SS` |

Laravel harus mengembalikan HTTP 2xx (mis. 200/201) agar Python menganggap kirim sukses.

---

## 5. Alur lengkap (Python + storage + Laravel)

1. **IP Webcam** streaming ke URL di `WEBCAM_URL`.
2. **Python** setiap `CAPTURE_INTERVAL_SEC` detik:
   - Ambil frame → simpan foto ke folder **storage** (nama file mis. `YYYYMMDD_HHMMSS.jpg`).
   - Deteksi kendaraan pada foto (ada/tidak, tipe, warna).
   - **Jika tidak ada kendaraan** dan **DELETE_IMAGE_IF_NO_VEHICLE = True**: hapus foto tersebut.
   - **Jika ada kendaraan**: simpan foto, tulis log analisis + update `latest_analysis.json`, dan (jika mode Laravel) POST ke Laravel `POST /api/detection`.
3. **Laravel** menerima POST → validasi → simpan/cache → isi form → response 2xx.
4. **Laravel / frontend** bisa baca hasil terbaru dari Python lewat **GET http://localhost:5000/latest** (format lihat bagian 1.2).

---

## 6. Prompt untuk update Laravel (receive detection)

```
Buat/update endpoint Laravel: POST /api/detection
- Menerima JSON: vehicle_type (car|motorcycle|truck|bus), color, confidence (0-1), timestamp.
- Validasi, simpan ke DB/cache, return 2xx.
- Untuk tampilkan deteksi terbaru tanpa menunggu POST, Laravel bisa polling GET http://localhost:5000/latest ke Python (field: photo, timestamp, detections).
```

---

## 7. Contoh implementasi Laravel (ringkas)

### routes/api.php

```php
Route::post('/detection', [DetectionController::class, 'store']);
```

### Controller: terima POST + contoh ambil data terbaru dari Python

```php
public function store(Request $request)
{
    $validated = $request->validate([
        'vehicle_type' => 'required|string|in:car,motorcycle,truck,bus',
        'color'        => 'required|string|max:50',
        'confidence'   => 'required|numeric|min:0|max:1',
        'timestamp'    => 'required|string',
    ]);
    Cache::put('latest_detection', $validated, 3600);
    return response()->json(['success' => true, 'data' => $validated], 201);
}

// Ambil hasil terbaru dari Python (GET /latest)
public function latestFromYolo()
{
    $response = Http::get('http://localhost:5000/latest');
    return response()->json($response->json());
}
```
