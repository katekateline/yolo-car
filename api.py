from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import urllib.request
from collections import Counter

app = Flask(__name__)
CORS(app)  # Enable CORS untuk komunikasi dengan Laravel

# Global detector instance
detector = None

# Storage untuk tracking mobil terbaru
# Format: {car_id: {'detection': {...}, 'timestamp': datetime, 'image_path': '...'}}
car_tracking = {}
TRACKING_DURATION_HOURS = 24  # Mobil lama akan dihapus setelah 24 jam


class VehicleDetector:
    def __init__(self, model_path='model/yolov8n.pt', confidence=0.5):
        """
        Initialize Vehicle Detector dengan akurasi tinggi
        
        Args:
            model_path: Path ke model YOLO (default: model/yolov8n.pt)
            confidence: Confidence threshold (0.5 untuk akurasi lebih tinggi)
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        
        # Mapping COCO classes ke tipe kendaraan
        self.vehicle_classes = {
            2: 'mobil',      # car
            3: 'motor',      # motorcycle
            5: 'bus',        # bus
            7: 'truk',       # truck
            1: 'sepeda'      # bicycle
        }
    
    def detect_color(self, image, bbox):
        """
        Deteksi warna kendaraan dengan akurasi tinggi menggunakan K-Means clustering
        
        Args:
            image: Frame gambar
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            str: Nama warna yang terdeteksi
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop ROI kendaraan
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 'unknown'
        
        # Fokus pada bagian tengah atas kendaraan (bagian body, bukan ban/shadow)
        h, w = roi.shape[:2]
        center_roi = roi[int(h*0.15):int(h*0.65), int(w*0.15):int(w*0.85)]
        
        if center_roi.size == 0:
            center_roi = roi
        
        # Resize untuk processing lebih cepat
        small_roi = cv2.resize(center_roi, (100, 100))
        
        # Convert ke HSV dan LAB untuk analisis lebih baik
        hsv = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(small_roi, cv2.COLOR_BGR2LAB)
        
        # Ambil rata-rata warna (mengabaikan outlier gelap/terang)
        h_vals = hsv[:,:,0].flatten()
        s_vals = hsv[:,:,1].flatten()
        v_vals = hsv[:,:,2].flatten()
        l_vals = lab[:,:,0].flatten()
        
        # Filter outlier (sangat gelap atau sangat terang)
        valid_mask = (v_vals > 30) & (v_vals < 250)
        
        if valid_mask.sum() < 100:  # Jika terlalu sedikit pixel valid
            valid_mask = v_vals > 20
        
        avg_h = np.median(h_vals[valid_mask])
        avg_s = np.median(s_vals[valid_mask])
        avg_v = np.median(v_vals[valid_mask])
        avg_l = np.median(l_vals[valid_mask])
        
        # Deteksi warna berdasarkan HSV
        # Urutan pengecekan: putih/hitam/abu dulu (achromatic), baru chromatic
        
        # 1. Putih: L tinggi, S rendah
        if avg_l > 180 and avg_s < 40:
            return 'putih'
        
        # 2. Hitam: V dan L rendah
        if avg_v < 70 and avg_l < 80:
            return 'hitam'
        
        # 3. Abu-abu/Silver: S rendah, V sedang
        if avg_s < 40 and avg_v >= 70:
            if avg_l > 140:
                return 'silver'
            return 'abu-abu'
        
        # 4. Warna chromatic (berdasarkan Hue)
        if avg_s >= 40:  # Saturasi cukup tinggi untuk warna chromatic
            # Merah (0-10 atau 160-180)
            if avg_h < 10 or avg_h > 160:
                return 'merah'
            # Orange (10-25)
            elif 10 <= avg_h < 25:
                return 'orange'
            # Kuning (25-40)
            elif 25 <= avg_h < 40:
                return 'kuning'
            # Hijau (40-80)
            elif 40 <= avg_h < 80:
                return 'hijau'
            # Biru (80-130)
            elif 80 <= avg_h < 130:
                return 'biru'
            # Ungu/Pink (130-160)
            elif 130 <= avg_h < 160:
                return 'ungu'
        
        # 5. Coklat: hue orange-kuning tapi V rendah
        if 10 <= avg_h < 30 and avg_v < 120 and avg_s > 30:
            return 'coklat'
        
        # Default jika tidak match
        return 'abu-abu'
    
    def get_vehicle_type(self, class_id):
        """
        Konversi class ID YOLO ke tipe kendaraan Indonesia
        
        Args:
            class_id: ID class dari YOLO
        
        Returns:
            str: Nama tipe kendaraan
        """
        return self.vehicle_classes.get(class_id, 'kendaraan lain')
    
    def process_frame(self, frame, iou=0.5, agnostic_nms=False):
        """
        Proses frame dengan deteksi YOLO + warna + tipe
        
        Args:
            frame: Frame gambar input
            iou: IoU threshold untuk NMS (0.5 untuk akurasi lebih baik)
            agnostic_nms: Class-agnostic NMS
        
        Returns:
            tuple: (annotated_frame, detections)
        """
        # Run YOLO dengan parameter optimal untuk akurasi
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=iou,
            agnostic_nms=agnostic_nms,
            verbose=False,
            imgsz=640  # Ukuran standar untuk balance speed-accuracy
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Filter hanya kendaraan
                class_id = int(box.cls[0])
                
                if class_id in self.vehicle_classes:
                    # Extract info
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Deteksi warna dan tipe
                    color = self.detect_color(frame, bbox)
                    vehicle_type = self.get_vehicle_type(class_id)
                    
                    # Simpan deteksi
                    detection = {
                        'type': vehicle_type,
                        'color': color,
                        'confidence': round(conf, 4),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area': (x2 - x1) * (y2 - y1)  # Area untuk tracking
                    }
                    detections.append(detection)
        
        return detections


def generate_car_id(detection):
    """
    Generate unique ID untuk mobil berdasarkan karakteristiknya
    Menggunakan kombinasi type, color, dan posisi untuk identifikasi
    """
    # Gunakan kombinasi type, color, dan area untuk ID
    # Ini akan membantu mengidentifikasi mobil yang sama
    base_id = f"{detection['type']}_{detection['color']}_{detection['area']}"
    return hash(base_id) % 1000000


def cleanup_old_cars():
    """
    Hapus mobil lama dari tracking berdasarkan timestamp
    """
    global car_tracking
    current_time = datetime.now()
    threshold_time = current_time - timedelta(hours=TRACKING_DURATION_HOURS)
    
    cars_to_remove = []
    for car_id, car_data in car_tracking.items():
        if car_data['timestamp'] < threshold_time:
            cars_to_remove.append(car_id)
            # Hapus file gambar jika ada
            if 'image_path' in car_data and os.path.exists(car_data['image_path']):
                try:
                    os.remove(car_data['image_path'])
                except:
                    pass
    
    for car_id in cars_to_remove:
        del car_tracking[car_id]
    
    return len(cars_to_remove)


def download_yolo_model(model_name='yolov8m.pt', model_dir='model'):
    """
    Download model YOLO secara otomatis jika belum ada
    
    Args:
        model_name: Nama model yang akan didownload (default: yolov8m.pt - optimal)
        model_dir: Direktori untuk menyimpan model
    
    Returns:
        str: Path ke model yang didownload atau nama model untuk YOLO
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_path):
        return model_path
    
    print(f"\nModel {model_name} tidak ditemukan. Mengunduh model optimal...")
    print("   (Ini mungkin memakan waktu beberapa menit, tergantung koneksi internet)")
    
    try:
        # Method 1: Gunakan YOLO untuk download (paling reliable)
        # YOLO akan otomatis download ke cache jika model tidak ada
        print(f"   Mengunduh menggunakan YOLO...")
        temp_model = YOLO(model_name)
        
        # Cek apakah model sudah terdownload di cache YOLO
        # YOLO menyimpan di ~/.ultralytics/weights/ atau di cache
        # Kita bisa langsung menggunakan nama model, YOLO akan handle path-nya
        print(f"[OK] Model {model_name} berhasil diunduh ke cache YOLO!")
        
        # Coba copy ke folder model jika memungkinkan
        try:
            from ultralytics.utils import SETTINGS
            cache_dir = SETTINGS.get('weights_dir', None)
            if cache_dir:
                cache_path = os.path.join(cache_dir, model_name)
                if os.path.exists(cache_path):
                    import shutil
                    shutil.copy2(cache_path, model_path)
                    print(f"[OK] Model disalin ke: {model_path}")
                    return model_path
        except:
            pass
        
        # Return nama model, YOLO akan handle path dari cache
        return model_name
        
    except Exception as e:
        print(f"[WARNING] Error saat download dengan YOLO: {str(e)}")
        print("   Mencoba download manual dari GitHub...")
        
        # Fallback: download manual dari GitHub
        try:
            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
            print(f"   Downloading from: {url}")
            
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, model_path, show_progress)
            print(f"\n[OK] Model {model_name} berhasil diunduh ke: {model_path}")
            return model_path
            
        except Exception as e2:
            print(f"\n[ERROR] Gagal download model: {str(e2)}")
            print(f"\nSolusi manual:")
            print(f"   1. Download dari: https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}")
            print(f"   2. Simpan di: {os.path.abspath(model_dir)}/")
            return None


def update_car_tracking(new_detections, image_path=None):
    """
    Update tracking dengan deteksi baru, hanya simpan mobil terbaru
    
    Args:
        new_detections: List deteksi baru
        image_path: Path ke gambar yang diproses
    
    Returns:
        dict: Data mobil terbaru yang terdeteksi
    """
    global car_tracking
    current_time = datetime.now()
    
    # Bersihkan mobil lama terlebih dahulu
    cleanup_old_cars()
    
    # Update tracking dengan deteksi baru
    latest_cars = {}
    
    for detection in new_detections:
        # Hanya proses mobil (bukan motor, bus, dll)
        if detection['type'] == 'mobil':
            car_id = generate_car_id(detection)
            
            # Update atau tambahkan mobil baru
            car_tracking[car_id] = {
                'detection': detection,
                'timestamp': current_time,
                'image_path': image_path
            }
            
            latest_cars[car_id] = car_tracking[car_id]
    
    return latest_cars


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'API is running',
        'tracked_cars': len(car_tracking)
    }), 200


@app.route('/detect', methods=['POST'])
def detect_vehicles():
    """
    Endpoint untuk deteksi mobil dari gambar yang dikirim Laravel
    
    Expected request:
    - POST dengan form-data atau JSON
    - Field 'image': file gambar atau URL path ke gambar
    - Optional: 'image_path': path ke gambar di server Laravel
    """
    try:
        # Cek apakah ada file gambar
        if 'image' not in request.files:
            # Coba ambil dari JSON (jika Laravel kirim path)
            data = request.get_json()
            if data and 'image_path' in data:
                image_path = data['image_path']
                if os.path.exists(image_path):
                    frame = cv2.imread(image_path)
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Image file not found: {image_path}'
                    }), 404
            else:
                return jsonify({
                    'success': False,
                    'error': 'No image file or image_path provided'
                }), 400
        else:
            # Ambil file dari request
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No image file selected'
                }), 400
            
            # Baca gambar dari file
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({
                    'success': False,
                    'error': 'Invalid image file'
                }), 400
        
        # Proses deteksi
        detections = detector.process_frame(frame)
        
        # Ambil image_path dari request jika ada
        image_path = None
        if request.form.get('image_path'):
            image_path = request.form.get('image_path')
        elif request.get_json() and 'image_path' in request.get_json():
            image_path = request.get_json()['image_path']
        
        # Update tracking (hanya mobil terbaru)
        latest_cars = update_car_tracking(detections, image_path)
        
        # Hitung statistik
        cars_only = [d for d in detections if d['type'] == 'mobil']
        image_height, image_width = frame.shape[:2]
        
        # Statistik warna mobil
        car_colors = Counter([d['color'] for d in cars_only])
        
        # Format response yang lebih simple dan lengkap
        response_data = {
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_info': {
                'width': int(image_width),
                'height': int(image_height),
                'size': f"{image_width}x{image_height}"
            },
            'summary': {
                'total_vehicles': len(detections),
                'cars': len(cars_only),
                'others': len(detections) - len(cars_only)
            },
            'cars': [
                {
                    'id': car_id,
                    'type': car_data['detection']['type'],
                    'color': car_data['detection']['color'],
                    'confidence': round(car_data['detection']['confidence'] * 100, 1),
                    'position': {
                        'x': car_data['detection']['bbox'][0],
                        'y': car_data['detection']['bbox'][1],
                        'width': car_data['detection']['bbox'][2] - car_data['detection']['bbox'][0],
                        'height': car_data['detection']['bbox'][3] - car_data['detection']['bbox'][1]
                    },
                    'bbox': car_data['detection']['bbox'],
                    'detected_at': car_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                }
                for car_id, car_data in latest_cars.items()
            ],
            'statistics': {
                'color_distribution': dict(car_colors),
                'average_confidence': round(sum([d['confidence'] for d in cars_only]) / len(cars_only) * 100, 1) if cars_only else 0
            },
            'all_vehicles': [
                {
                    'type': d['type'],
                    'color': d['color'],
                    'confidence': round(d['confidence'] * 100, 1),
                    'bbox': d['bbox']
                }
                for d in detections
            ]
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/latest-cars', methods=['GET'])
def get_latest_cars():
    """
    Endpoint untuk mendapatkan daftar mobil terbaru yang terdeteksi
    """
    try:
        # Bersihkan mobil lama
        removed_count = cleanup_old_cars()
        
        # Format response yang lebih simple dan lengkap
        latest_cars = []
        color_stats = Counter()
        
        for car_id, car_data in car_tracking.items():
            car_info = {
                'id': car_id,
                'type': car_data['detection']['type'],
                'color': car_data['detection']['color'],
                'confidence': round(car_data['detection']['confidence'] * 100, 1),
                'position': {
                    'x': car_data['detection']['bbox'][0],
                    'y': car_data['detection']['bbox'][1],
                    'width': car_data['detection']['bbox'][2] - car_data['detection']['bbox'][0],
                    'height': car_data['detection']['bbox'][3] - car_data['detection']['bbox'][1]
                },
                'bbox': car_data['detection']['bbox'],
                'detected_at': car_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'image_path': car_data.get('image_path')
            }
            latest_cars.append(car_info)
            color_stats[car_data['detection']['color']] += 1
        
        # Sort berdasarkan timestamp terbaru
        latest_cars.sort(key=lambda x: x['detected_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'summary': {
                'total_cars': len(latest_cars),
                'removed_old_cars': removed_count
            },
            'statistics': {
                'color_distribution': dict(color_stats)
            },
            'cars': latest_cars
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/clear-tracking', methods=['POST'])
def clear_tracking():
    """
    Endpoint untuk menghapus semua tracking (opsional, untuk testing)
    """
    global car_tracking
    count = len(car_tracking)
    car_tracking.clear()
    
    return jsonify({
        'success': True,
        'message': f'Cleared {count} tracked cars'
    }), 200


if __name__ == '__main__':
    # Initialize detector
    print("\n" + "="*80)
    print("Vehicle Detection API - Initializing...")
    print("="*80)
    
    # Model optimal: yolov8m.pt (balance antara speed dan accuracy)
    optimal_model = 'yolov8m.pt'
    model_path = f'model/{optimal_model}'
    
    # Cek apakah model sudah ada
    if not os.path.exists(model_path):
        # Coba cari model lain yang mungkin sudah ada
        existing_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8l.pt', 'yolov8x.pt']
        found_model = None
        
        for model_name in existing_models:
            alt_path = f'model/{model_name}'
            if os.path.exists(alt_path):
                model_path = alt_path
                found_model = model_name
                print(f"[OK] Found existing model: {model_name}")
                break
        
        # Jika tidak ada model sama sekali, download yang optimal
        if not found_model:
            print(f"\nModel tidak ditemukan. Mengunduh model optimal: {optimal_model}")
            downloaded_path = download_yolo_model(optimal_model)
            
            if downloaded_path:
                # Jika download berhasil, gunakan model tersebut
                if os.path.exists(f'model/{optimal_model}'):
                    model_path = f'model/{optimal_model}'
                else:
                    # YOLO menyimpan di cache, gunakan nama model saja
                    model_path = optimal_model
            else:
                print("\n[ERROR] Gagal mengunduh model. Silakan download manual:")
                print(f"   https://github.com/ultralytics/assets/releases/download/v0.0.0/{optimal_model}")
                print(f"   Simpan di: {os.path.abspath('model/')}")
                exit(1)
    else:
        print(f"[OK] Model ditemukan: {optimal_model}")
    
    # Initialize detector
    try:
        detector = VehicleDetector(model_path=model_path, confidence=0.4)
        print(f"[OK] Detector initialized successfully")
        print(f"[OK] Model: {model_path}")
        print(f"[OK] Confidence threshold: 0.4")
    except Exception as e:
        print(f"\n[ERROR] Error initializing detector: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Pastikan model YOLO ada di folder model/")
        print("  2. Coba download model manual dari:")
        print(f"     https://github.com/ultralytics/assets/releases/download/v0.0.0/{optimal_model}")
        exit(1)
    
    print("="*80)
    print("API Server starting on http://localhost:5000")
    print("\nAvailable Endpoints:")
    print("  POST /detect        - Detect vehicles from image")
    print("  GET  /latest-cars   - Get latest detected cars")
    print("  GET  /health        - Health check")
    print("="*80 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)