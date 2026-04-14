# Integrasi Laravel ↔ YOLO Service (upload gambar → analisis → balikan Laravel)

Arsitektur utama: **Laravel** mengambil foto (tombol / kamera / upload file), mengirim gambar ke **Python (Flask)** lewat **POST multipart**, menerima **JSON hasil deteksi** dalam response yang sama. Hasil sekarang mencakup:

- deteksi kendaraan (`detections`)
- plat nomor (`plate_number`, opsional)
- **QR code** (`qr_codes` + `qr_texts`)

Opsional: Python **juga** mem-**POST** callback ke endpoint Laravel (server-to-server) jika diaktifkan.

Alur lama **IP Webcam di sisi Python** tetap didukung lewat env `ENABLE_WEBCAM=1` (lihat bagian akhir).

---

## 1. Ringkasan endpoint API Python (YOLO Service)

Default base URL: **http://127.0.0.1:5000** (ubah dengan env `PORT`).

| Method | Path      | Deskripsi |
|--------|-----------|-----------|
| GET    | `/status` | Health check |
| GET    | `/latest` | Snapshot hasil analisis terakhir (dari file `latest_analysis.json`) |
| POST   | `/analyze` | **Utama:** kirim file gambar (`multipart`), terima JSON hasil deteksi |

---

## 2. POST /analyze (wajib untuk alur Laravel → Python)

Laravel mengirim **multipart/form-data** dengan field file bernama **`image`** (satu foto JPG/PNG) dan field teks **`scan_type`**.

| Item   | Nilai |
|--------|--------|
| Method | `POST` |
| URL    | `http://127.0.0.1:5000/analyze` |
| Body   | `multipart/form-data` |
| Field `image` | File gambar (JPG/PNG) |
| Field `scan_type` | `vehicle` (default) atau `qr` |
| Header opsional | `X-API-Key: <secret>` — hanya jika di Python di-set env `YOLO_SERVICE_API_KEY` |

### Alur Kerja Berdasarkan `scan_type`:

1.  **`scan_type = vehicle` (Default)**:
    -   Python mengecek apakah ada kendaraan menggunakan model YOLO.
    -   Jika ada, Python mengidentifikasi **warna** dan **jenis kendaraan**.
    -   Setelah itu, Python mengirim potongan gambar kendaraan ke **PlateRecognizer API** untuk mendapatkan nomor plat.
    -   Hasil lengkap dikirim kembali ke Laravel.
2.  **`scan_type = qr`**:
    -   Python hanya melakukan scan **QR Code** pada gambar.
    -   Tidak melakukan deteksi kendaraan atau plat nomor.
    -   Hasil QR dikirim kembali ke Laravel.

### Response 200 (sukses) — contoh ada kendaraan (`scan_type=vehicle`)

```json
{
  "success": true,
  "source": "yolo_analyze_upload",
  "timestamp": "2026-04-08 14:30:00",
  "updated_at": "2026-04-08T14:30:00.123456",
  "photo": "storage/20260408_143000_a1b2c3d4.jpg",
  "detections": [
    {
      "vehicle_type": "car",
      "color": "hitam",
      "confidence": 0.8734,
      "bbox": [100, 200, 400, 500]
    }
  ],
  "qr_codes": [
    {
      "text": "PARKING-A1-2026",
      "points": [[12, 34], [120, 30], [122, 140], [14, 142]]
    }
  ],
  "qr_texts": ["PARKING-A1-2026"],
  "plate_number": "B1234XYZ",
  "vehicle_type": "car",
  "color": "hitam",
  "confidence": 0.87
}
```

Field **`vehicle_type`**, **`color`**, **`confidence`** di root menyalin **kendaraan pertama** (kompatibel dengan endpoint Laravel lama). **`detections`** berisi semua kendaraan terdeteksi pada gambar.

Field QR:

- `qr_codes`: array detail QR (`text` + `points` sudut QR jika tersedia).
- `qr_texts`: array string praktis untuk dipakai langsung di form/validasi Laravel.

### Response 200 — tidak ada kendaraan

```json
{
  "success": true,
  "timestamp": "2026-04-08 14:30:00",
  "updated_at": "2026-04-08T14:30:00.123456",
  "photo": null,
  "detections": [],
  "qr_codes": [],
  "qr_texts": [],
  "plate_number": null,
  "vehicle_type": null,
  "color": null,
  "confidence": null,
  "message": "no_vehicle"
}
```

Jika opsi hapus foto tanpa kendaraan aktif di Python, `photo` akan `null`.

### Error umum

| HTTP | `error`           | Keterangan |
|------|-------------------|------------|
| 400  | `missing_file`    | Field `image` tidak ada |
| 400  | `invalid_image`   | Bukan gambar yang valid |
| 401  | `unauthorized`    | API key salah / tidak ada |
| 500  | `detection_failed`| Gagal inferensi (lihat `message`) |

---

## 3. Callback opsional dari Python ke Laravel (server-to-server)

Jika Laravel ingin menerima data **tanpa** hanya mengandalkan response HTTP dari request upload (misalnya worker lain), set di lingkungan Python:

- `LARAVEL_CALLBACK_AFTER_ANALYZE=true`

Maka setelah analisis, Python akan **POST JSON** yang sama (struktur mirip response di atas) ke **`LARAVEL_URL`**.

Hindari mengaktifkan callback jika controller Laravel yang memanggil `/analyze` **juga** menyimpan hasil dari **body response** — bisa dobel pemrosesan.

---

## 4. GET /status dan GET /latest

Sama seperti sebelumnya: `GET /status` → `{"status":"running"}`; `GET /latest` mengembalikan isi analisis terakhir yang tersimpan di disk (berguna untuk polling/debug).

---

## 5. Kontrak API Laravel (yang disediakan tim Laravel)

### 5.1 Endpoint untuk menerima deteksi (opsional / legacy)

**POST** `http://localhost:8000/api/detection` (sesuaikan domain; di Python set env `LARAVEL_URL`)

Digunakan jika:

- Mode **IP Webcam** Python (`ENABLE_WEBCAM=1`) dengan `LARAVEL_MODE=true`, atau
- **`LARAVEL_CALLBACK_AFTER_ANALYZE=true`** setelah `/analyze`.

**Headers:** `Content-Type: application/json`

**Body (minimal — satu kendaraan / callback sederhana):**

```json
{
  "vehicle_type": "car",
  "color": "hitam",
  "confidence": 0.87,
  "timestamp": "2026-04-08 14:00:00",
  "plate_number": "B1234XYZ",
  "qr_codes": [
    { "text": "PARKING-A1-2026", "points": [[12, 34], [120, 30], [122, 140], [14, 142]] }
  ],
  "qr_texts": ["PARKING-A1-2026"]
}
```

**Body (callback dari `/analyze` — lengkap):** bisa menyertakan `success`, `detections`, `photo`, `source`, `updated_at`, `plate_number`, `qr_codes`, `qr_texts`, dll. seperti response `/analyze`. Tim Laravel disarankan menerima **JSON fleksibel** dan menyimpan `detections` + data QR.

---

## 6. Contoh Laravel: tombol ambil gambar → kirim ke Python → pakai hasil

### 6.1 Route

```php
Route::post('/parking/analyze-frame', [DetectionController::class, 'analyzeFrame']);
```

### 6.2 Controller (multipart `image`)

```php
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

public function analyzeFrame(Request $request)
{
    $request->validate([
        'image' => 'required|image|max:10240',
        'scan_type' => 'nullable|in:vehicle,qr',
    ]);

    $path = $request->file('image')->getRealPath();
    $scanType = $request->input('scan_type', 'vehicle');

    $http = Http::timeout(120);

    $response = $http->attach(
        'image',
        file_get_contents($path),
        $request->file('image')->getClientOriginalName()
    )->post('http://127.0.0.1:5000/analyze', [
        'scan_type' => $scanType
    ]);

    if (!$response->successful()) {
        return response()->json(['error' => 'YOLO service error', 'body' => $response->body()], 502);
    }

    return response()->json($response->json());
}
```

### 6.3 Blade (ringkas): input file + submit

```html
<form action="{{ route('parking.analyze-frame') }}" method="post" enctype="multipart/form-data">
    @csrf
    <input type="file" name="image" accept="image/*" capture="environment" required>
    <button type="submit">Analisis</button>
</form>
```

Atribut `capture` membantu perangkat mobile membuka kamera; untuk desktop tetap bisa pilih file.

---

## 7. Environment variables (Python)

| Variable | Keterangan |
|----------|------------|
| `LARAVEL_URL` | URL endpoint Laravel untuk callback (default `http://localhost:8000/api/detection`) |
| `LARAVEL_CALLBACK_AFTER_ANALYZE` | `true` / `false` — POST hasil ke `LARAVEL_URL` setelah `/analyze` |
| `PLATERECOGNIZER_TOKEN` | Token PlateRecognizer (opsional); tanpa token, plat tidak dibaca |
| `YOLO_SERVICE_API_KEY` | Jika di-set, request `/analyze` wajib header `X-API-Key` |
| `CORS_ALLOW_ORIGIN` | Default `*` (untuk browser yang memanggil langsung) |
| `PORT` | Port Flask (default `5000`) |
| `ENABLE_WEBCAM` | `1` = aktifkan loop IP Webcam di Python (default mati) |
| `WEBCAM_URL` | URL stream jika `ENABLE_WEBCAM=1` |
| `LARAVEL_MODE` | Bersama `ENABLE_WEBCAM`: kirim POST ke Laravel saat ada kendaraan dari webcam |
| `YOLO_MODEL` | Override nama/path model (default `yolov8m.pt`, unduh otomatis jika belum ada) |
| `CLI_CAPTURE_ONLY` | `1` = hanya loop webcam tanpa Flask (jarang dipakai) |

---

## 8. Setup Python

```bash
cd yolo-service
pip install -r requirements.txt
python app.py
```

Model **yolov8m.pt**: jika belum ada di folder project, **Ultralytics akan mengunduh otomatis** saat pertama kali model dimuat (perlu internet). Alternatif: taruh `yolov8m.pt` di root repo atau `best.pt` di folder `yolo-service`.

---

## 9. Prompt singkat untuk tim Laravel (salin)

```
Implementasi:
- Form upload foto (atau capture kamera) → POST multipart ke backend Laravel.
- Backend Laravel forward file ke Python POST http://127.0.0.1:5000/analyze dengan field "image".
- Baca JSON response: success, detections[], vehicle_type, color, confidence, plate_number, qr_codes[], qr_texts[], photo.
- Simpan ke database / tampilkan di UI.
- Opsional: sediakan POST /api/detection yang menerima JSON callback jika Python di-set LARAVEL_CALLBACK_AFTER_ANALYZE=true.
```

---

## 10. Alur lengkap (disarankan)

1. User klik tombol di Laravel → foto di-upload ke Laravel.
2. Laravel memanggil **POST /analyze** ke Python dengan field **`image`**.
3. Python mengembalikan **JSON** (kendaraan + plat + QR code) dalam response → Laravel menyimpan/menampilkan.
4. (Opsional) Python mem-POST ulang ke Laravel jika `LARAVEL_CALLBACK_AFTER_ANALYZE=true`.

Alur lama: Python polling **IP Webcam** tetap bisa dengan `ENABLE_WEBCAM=1` dan dokumentasi env di atas.
