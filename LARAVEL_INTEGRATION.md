# Integrasi Laravel dengan Python Car Detection API

## Setup Python API

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Pastikan model YOLO ada di folder `model/`:
   - `model/yolov8n.pt` (atau model lainnya)

3. Jalankan API:
```bash
python api.py
```

API akan berjalan di `http://localhost:5000`

## Endpoints API

### 1. POST /detect
Deteksi mobil dari gambar yang diupload Laravel

**Request (Form Data):**
```
POST http://localhost:5000/detect
Content-Type: multipart/form-data

image: [file gambar]
image_path: [optional] path ke gambar di Laravel (contoh: public/images/car.jpg)
```

**Request (JSON):**
```json
{
  "image_path": "public/images/car.jpg"
}
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-02-13T10:30:00",
  "total_detections": 3,
  "cars_detected": 2,
  "latest_cars": [
    {
      "id": 123456,
      "type": "mobil",
      "color": "merah",
      "confidence": 0.85,
      "bbox": [100, 150, 300, 400],
      "detected_at": "2026-02-13T10:30:00"
    }
  ],
  "all_detections": [...]
}
```

### 2. GET /latest-cars
Mendapatkan daftar mobil terbaru yang terdeteksi (maksimal 24 jam terakhir)

**Response:**
```json
{
  "success": true,
  "total_cars": 5,
  "removed_old_cars": 2,
  "cars": [...]
}
```

### 3. GET /health
Health check endpoint

## Contoh Kode Laravel

### Controller Example

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;

class CarDetectionController extends Controller
{
    private $pythonApiUrl = 'http://localhost:5000';
    
    public function detectCar(Request $request)
    {
        // Validasi request
        $request->validate([
            'image' => 'required|image|mimes:jpeg,png,jpg|max:10240'
        ]);
        
        // Simpan gambar ke public/images
        $imagePath = $request->file('image')->store('images', 'public');
        $fullPath = storage_path('app/public/' . $imagePath);
        
        try {
            // Kirim gambar ke Python API
            $response = Http::attach(
                'image', file_get_contents($fullPath), basename($fullPath)
            )->post($this->pythonApiUrl . '/detect', [
                'image_path' => 'public/' . $imagePath
            ]);
            
            if ($response->successful()) {
                $data = $response->json();
                
                // Simpan hasil deteksi ke database jika perlu
                // ...
                
                return response()->json([
                    'success' => true,
                    'data' => $data,
                    'image_path' => $imagePath
                ]);
            } else {
                return response()->json([
                    'success' => false,
                    'error' => 'Detection failed'
                ], 500);
            }
            
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }
    
    public function getLatestCars()
    {
        try {
            $response = Http::get($this->pythonApiUrl . '/latest-cars');
            
            if ($response->successful()) {
                return response()->json($response->json());
            }
            
            return response()->json([
                'success' => false,
                'error' => 'Failed to fetch latest cars'
            ], 500);
            
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }
}
```

### Routes Example (routes/web.php atau routes/api.php)

```php
use App\Http\Controllers\CarDetectionController;

// Web routes
Route::post('/detect-car', [CarDetectionController::class, 'detectCar']);
Route::get('/latest-cars', [CarDetectionController::class, 'getLatestCars']);

// API routes
Route::prefix('api')->group(function () {
    Route::post('/detect-car', [CarDetectionController::class, 'detectCar']);
    Route::get('/latest-cars', [CarDetectionController::class, 'getLatestCars']);
});
```

### Frontend Example (Blade atau Vue/React)

```html
<!-- Form upload gambar -->
<form id="detectForm" enctype="multipart/form-data">
    @csrf
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Detect Car</button>
</form>

<script>
document.getElementById('detectForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    
    try {
        const response = await fetch('/detect-car', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            console.log('Cars detected:', data.data.latest_cars);
            // Tampilkan hasil di UI
        }
    } catch (error) {
        console.error('Error:', error);
    }
});
</script>
```

## Fitur Tracking Mobil Terbaru

- API secara otomatis hanya menyimpan deteksi mobil terbaru
- Mobil lama (lebih dari 24 jam) akan otomatis dihapus
- Setiap mobil memiliki ID unik berdasarkan karakteristiknya
- Data tracking disimpan di memory (akan reset saat API restart)

## Catatan Penting

1. Pastikan Python API berjalan sebelum Laravel mengirim request
2. Pastikan CORS sudah diaktifkan (sudah termasuk di api.py)
3. Ukuran gambar maksimal disarankan 10MB
4. Format gambar yang didukung: JPEG, PNG, JPG
5. Model YOLO harus ada di folder `model/` sebelum menjalankan API
