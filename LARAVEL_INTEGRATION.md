# Integrasi Laravel dengan YOLO Service (IP Webcam → Python → Laravel)

Arsitektur baru: **Python** mengambil stream dari **IP Webcam**, mendeteksi kendaraan (YOLO + warna), lalu **mengirim data ke Laravel** lewat `POST /api/detection`. Laravel hanya **menerima** JSON dan mengisi form (tidak lagi mengupload gambar ke Python).

---

## 1. Setup Python (YOLO Service)

### 1.1 Install dependencies

```bash
cd yolo_service
pip install -r requirements.txt
```

### 1.2 Model YOLO

Gunakan model **yolov8m.pt** di **folder root project** (satu tingkat di atas `yolo_service/`):

```
yolo-car/
├── yolov8m.pt    ← taruh di sini
└── yolo_service/
    ├── app.py
    ├── detector.py
    ├── color_detector.py
    └── requirements.txt
```

Atau taruh **best.pt** di dalam `yolo_service/` jika punya model custom. Prioritas: `yolov8m.pt` (root) → `best.pt` (yolo_service) → default.

### 1.3 Konfigurasi

Edit `yolo_service/app.py` atau set env:

- **WEBCAM_URL** – URL stream IP Webcam, contoh: `http://192.168.1.100:8080/video`
- **LARAVEL_URL** – URL endpoint Laravel: `http://localhost:8000/api/detection`

```bash
# Windows (PowerShell)
$env:WEBCAM_URL="http://IP_HP:8080/video"
$env:LARAVEL_URL="http://localhost:8000/api/detection"

# Linux/Mac
export WEBCAM_URL="http://IP_HP:8080/video"
export LARAVEL_URL="http://localhost:8000/api/detection"
```

### 1.4 Jalankan service

```bash
cd yolo_service
python app.py
```

Service berjalan di `http://localhost:5000`. Detection berjalan otomatis di background (bukan lewat Flask).

### 1.5 Health check

```bash
GET http://localhost:5000/status
```

Response:

```json
{
  "status": "running"
}
```

---

## 2. Kontrak API Laravel (yang harus disediakan Laravel)

Python akan **memanggil** endpoint ini setiap kali ada kendaraan terdeteksi (dengan cooldown 3 detik):

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

| Field          | Tipe    | Keterangan |
|----------------|---------|------------|
| `vehicle_type` | string  | `car`, `motorcycle`, `truck`, `bus` |
| `color`        | string  | `merah`, `biru`, `hitam`, `putih`, `abu`, `silver`, `hijau`, `kuning`, `orange`, `ungu`, `coklat` |
| `confidence`   | float   | 0–1 (confidence YOLO) |
| `timestamp`    | string  | Format `YYYY-MM-DD HH:MM:SS` |

**Laravel harus:**

- Menerima POST tersebut (route + controller).
- Validasi request (vehicle_type, color, confidence, timestamp).
- Simpan ke database / gunakan untuk auto-fill form (sesuai kebutuhan).
- Return HTTP 2xx (misal 200 atau 201) agar Python menganggap kirim sukses.

---

## 3. Prompt untuk Update Laravel (copy-paste ke AI / developer)

Gunakan blok di bawah ini sebagai **instruksi untuk mengupdate sisi Laravel** agar cocok dengan YOLO service yang baru.

---

```
## Konteks
Ada service Python (YOLO) yang mendeteksi kendaraan dari IP Webcam dan mengirim data ke Laravel via POST. Laravel tidak lagi mengupload gambar ke Python. Yang dibutuhkan:

1. Buat/update endpoint Laravel: **POST /api/detection**
2. Endpoint ini **hanya menerima** JSON dari Python (bukan upload gambar).

## Request dari Python (POST /api/detection)

- Method: POST
- Content-Type: application/json
- Body contoh:
{
  "vehicle_type": "car",
  "color": "hitam",
  "confidence": 0.87,
  "timestamp": "2026-02-22 14:00:00"
}

- vehicle_type: string, nilai yang mungkin: "car", "motorcycle", "truck", "bus"
- color: string, contoh: "merah", "biru", "hitam", "putih", "abu", "silver", "hijau", "kuning", "orange", "ungu", "coklat"
- confidence: float 0-1
- timestamp: string format "Y-m-d H:i:s"

## Yang harus dilakukan Laravel

1. **Route:** tambah POST route ke /api/detection (di routes/api.php atau yang dipakai project).
2. **Controller:** satu method yang:
   - Menerima request JSON (vehicle_type, color, confidence, timestamp).
   - Validasi field tersebut (required, vehicle_type in [car, motorcycle, truck, bus], color string, confidence numeric 0-1, timestamp string).
   - Simpan ke database (misal tabel detections atau vehicles) ATAU gunakan data untuk auto-fill form (session, broadcast ke frontend, dll).
   - Return JSON response sukses (status 200 atau 201). Contoh: { "success": true, "message": "Detection received" }.
3. **Tidak perlu:** endpoint untuk upload gambar ke Python, GET /latest-cars ke Python, atau panggilan dari Laravel ke Python untuk deteksi. Python yang memanggil Laravel.
4. **CORS:** pastikan Laravel mengizinkan request dari origin Python jika beda origin (atau Python di localhost:5000, Laravel di localhost:8000 biasanya tidak kena CORS untuk POST dari server-side; kalau ada CORS error, tambah origin yang perlu di Laravel CORS config).
5. **Auto-fill form:** jika ada form input (tipe kendaraan, warna, dll), isi otomatis dari data terakhir yang diterima (bisa simpan di session/cache/DB dan baca di halaman form).
```

---

## 4. Contoh Implementasi Laravel (ringkas)

### routes/api.php

```php
use App\Http\Controllers\DetectionController;

Route::post('/detection', [DetectionController::class, 'store']);
```

### App\Http\Controllers\DetectionController.php

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache; // atau DB / Session

class DetectionController extends Controller
{
    public function store(Request $request)
    {
        $validated = $request->validate([
            'vehicle_type' => 'required|string|in:car,motorcycle,truck,bus',
            'color'        => 'required|string|max:50',
            'confidence'   => 'required|numeric|min:0|max:1',
            'timestamp'    => 'required|string',
        ]);

        // Contoh: simpan ke cache 1 jam untuk auto-fill form
        Cache::put('latest_detection', $validated, 3600);

        // Atau simpan ke database
        // Detection::create($validated);

        return response()->json([
            'success' => true,
            'message' => 'Detection received',
            'data'    => $validated,
        ], 201);
    }
}
```

### Auto-fill form (contoh di controller form)

```php
$latest = Cache::get('latest_detection');
// Pass $latest ke view, isi field: vehicle_type, color, confidence, timestamp
```

---

## 5. Alur Lengkap

1. **IP Webcam** (HP) streaming ke URL yang di-set di `WEBCAM_URL`.
2. **Python (yolo_service)** jalan di background: baca frame → setiap 5 frame jalankan YOLO (model **yolov8m.pt** di root project) → filter car/motorcycle/truck/bus → deteksi warna → cooldown 3 detik → **POST** ke Laravel `POST /api/detection`.
3. **Laravel** terima JSON → validasi → simpan/cache → isi form → response 2xx.

Tidak ada lagi flow: Laravel upload gambar → Python /detect. Semua deteksi dari stream kamera dan Python yang push ke Laravel.
