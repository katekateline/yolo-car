"""
YOLO Service: baca stream IP Webcam -> deteksi kendaraan -> kirim ke Laravel.
Flask hanya untuk GET /status. Detection berjalan di background thread.
"""
import threading
import time
import cv2
import requests
from datetime import datetime
from flask import Flask, jsonify

from detector import VehicleDetector

app = Flask(__name__)

# Konfigurasi (bisa diganti via env atau config)
WEBCAM_URL = "http://localhost:8080/video"  # Ganti dengan IP HP
LARAVEL_URL = "http://park-it.test/api/detection"
# Model: gunakan yolov8m.pt di root project, fallback best.pt lalu default
MODEL_PATH = "best.pt"
MODEL_ROOT = "yolov8m.pt"  # file di folder root project (parent dari yolo_service)
COOLDOWN_SEC = 3
PROCESS_EVERY_N_FRAMES = 5

# State global
_detector = None
_last_sent_time = 0
_running = False
_frame_count = 0
_reconnect_delay = 2
_max_reconnect_delay = 30


def ensure_detector():
    global _detector
    if _detector is None:
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_root_path = os.path.join(root_dir, MODEL_ROOT)
        model_local_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if os.path.exists(model_root_path):
            _detector = VehicleDetector(model_path=model_root_path)
            print(f"[INFO] Model dipakai: {model_root_path}")
        elif os.path.exists(model_local_path):
            _detector = VehicleDetector(model_path=model_local_path)
            print(f"[INFO] Model dipakai: {model_local_path}")
        else:
            _detector = VehicleDetector(model_path=MODEL_PATH)
            print(f"[INFO] Model dipakai: {MODEL_PATH} (default)")
    return _detector


def send_to_laravel(vehicle_type, color, confidence):
    """POST ke Laravel. Tidak crash jika Laravel tidak aktif."""
    global _last_sent_time
    data = {
        "vehicle_type": vehicle_type,
        "color": color,
        "confidence": round(confidence, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        r = requests.post(LARAVEL_URL, json=data, timeout=3)
        if r.ok:
            print("[INFO] Data dikirim ke Laravel")
        else:
            print(f"[ERROR] Laravel return status {r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gagal kirim data: {e}")


def detection_loop():
    """Loop utama: baca frame -> setiap N frame deteksi -> cooldown -> kirim ke Laravel."""
    global _running, _last_sent_time, _frame_count, _reconnect_delay
    cap = None
    current_delay = _reconnect_delay

    while _running:
        if cap is None or not cap.isOpened():
            print(f"[INFO] Menghubungkan ke webcam: {WEBCAM_URL}")
            cap = cv2.VideoCapture(WEBCAM_URL)
            if not cap.isOpened():
                print(f"[ERROR] Gagal buka stream, retry dalam {current_delay}s...")
                time.sleep(current_delay)
                current_delay = min(current_delay + 2, _max_reconnect_delay)
                continue
            current_delay = _reconnect_delay

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cap = None
            print("[WARN] Frame gagal diambil, reconnect...")
            time.sleep(current_delay)
            continue

        _frame_count += 1
        if _frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        try:
            det = ensure_detector()
            detections = det.process_frame(frame, process_vehicle_only=True)
        except Exception as e:
            print(f"[ERROR] Deteksi error: {e}")
            continue

        if not detections:
            continue

        now = time.time()
        for d in detections:
            if now - _last_sent_time < COOLDOWN_SEC:
                continue
            print(f"[INFO] Kendaraan terdeteksi | {d['vehicle_type']} | Warna: {d['color']} | Conf: {d['confidence']:.2f}")
            send_to_laravel(d["vehicle_type"], d["color"], d["confidence"])
            _last_sent_time = now
            break  # Satu kendaraan per cooldown window


def run_detection_background():
    """Jalankan detection loop di thread terpisah (satu thread saja)."""
    global _running
    _running = True
    thread = threading.Thread(target=detection_loop, daemon=True)
    thread.start()
    print("[INFO] Background detection thread started.")


@app.route("/status", methods=["GET"])
def status():
    """Health check. Detection tidak di-trigger dari sini."""
    return jsonify({"status": "running"})


if __name__ == "__main__":
    import os
    if os.environ.get("WEBCAM_URL"):
        WEBCAM_URL = os.environ.get("WEBCAM_URL")
    if os.environ.get("LARAVEL_URL"):
        LARAVEL_URL = os.environ.get("LARAVEL_URL")

    run_detection_background()
    print("[INFO] Flask /status available. Detection berjalan otomatis di background.")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
