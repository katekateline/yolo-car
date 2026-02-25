"""
YOLO Service: IP Webcam -> ambil foto tiap 5 detik -> simpan di storage -> deteksi kendaraan (tipe + warna).
"""
import threading
import time
import json
import os

import cv2
import requests
from datetime import datetime
from flask import Flask, jsonify

from detector import VehicleDetector

app = Flask(__name__)

# Konfigurasi
WEBCAM_URL = "http://localhost:8080/video"
LARAVEL_URL = "http://park-it.test/api/detection"

# Mode: False = CLI (output CMD saja), True = Laravel (POST + Flask)
LARAVEL_MODE = False

# Ambil foto dari webcam setiap N detik, simpan di folder storage
CAPTURE_INTERVAL_SEC = 5
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# True = hapus foto jika tidak ada kendaraan terdeteksi; False = simpan semua foto
DELETE_IMAGE_IF_NO_VEHICLE = True

# Model YOLO
MODEL_PATH = "best.pt"
MODEL_ROOT = "yolov8m.pt"
CONFIDENCE_THRESHOLD = 0.25
COOLDOWN_SEC = 3

# File JSON
ANALYSIS_LOG_PATH = os.path.join(os.path.dirname(__file__), "frame_analysis.jsonl")
LATEST_ANALYSIS_PATH = os.path.join(os.path.dirname(__file__), "latest_analysis.json")

# State global
_detector = None
_last_sent_time = 0
_running = False
_reconnect_delay = 2
_max_reconnect_delay = 30


def ensure_storage():
    """Buat folder storage jika belum ada."""
    os.makedirs(STORAGE_DIR, exist_ok=True)


def ensure_detector():
    global _detector
    if _detector is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_root_path = os.path.join(root_dir, MODEL_ROOT)
        model_local_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if os.path.exists(model_root_path):
            _detector = VehicleDetector(model_path=model_root_path, confidence=CONFIDENCE_THRESHOLD)
            print(f"[INFO] Model: {model_root_path}")
        elif os.path.exists(model_local_path):
            _detector = VehicleDetector(model_path=model_local_path, confidence=CONFIDENCE_THRESHOLD)
            print(f"[INFO] Model: {model_local_path}")
        else:
            _detector = VehicleDetector(model_path=MODEL_PATH, confidence=CONFIDENCE_THRESHOLD)
            print(f"[INFO] Model: {MODEL_PATH} (default)")
    return _detector


def save_frame(frame, filename_base=None):
    """
    Simpan frame sebagai foto di folder storage.
    Returns: (path_abs, path_rel) atau (None, None) jika gagal.
    """
    ensure_storage()
    if filename_base is None:
        filename_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_base}.jpg"
    path = os.path.join(STORAGE_DIR, filename)
    try:
        cv2.imwrite(path, frame)
        return path, os.path.join("storage", filename)
    except Exception as e:
        print(f"[WARN] Gagal simpan foto: {e}")
        return None, None


def delete_image(path_abs):
    """Hapus file gambar jika ada. Returns True jika berhasil atau file tidak ada."""
    if not path_abs:
        return True
    try:
        if os.path.isfile(path_abs):
            os.remove(path_abs)
            return True
    except Exception as e:
        print(f"[WARN] Gagal hapus gambar: {e}")
    return False


def log_frame_analysis(photo_path, detections, timestamp_str=None):
    """Append hasil analisis ke frame_analysis.jsonl."""
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "photo": photo_path,
        "timestamp": timestamp_str,
        "detections": detections,
    }
    try:
        with open(ANALYSIS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Gagal tulis log: {e}")


def update_latest_analysis(photo_path, detections, timestamp_str=None):
    """Overwrite latest_analysis.json dengan hasil terbaru."""
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "photo": photo_path,
        "timestamp": timestamp_str,
        "updated_at": datetime.now().isoformat(),
        "detections": detections,
    }
    try:
        with open(LATEST_ANALYSIS_PATH, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Gagal update latest: {e}")


def send_to_laravel(vehicle_type, color, confidence):
    """POST ke Laravel."""
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
            print(f"[ERROR] Laravel status {r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gagal kirim: {e}")


def capture_and_detect():
    """
    Loop utama:
    1. Baca frame dari IP Webcam
    2. Setiap CAPTURE_INTERVAL_SEC detik: simpan foto ke storage
    3. Deteksi kendaraan pada foto itu (ada/tidak, tipe, warna)
    4. Jika ada: log + update JSON (+ kirim Laravel jika mode Laravel)
    """
    global _running, _last_sent_time
    cap = None
    current_delay = _reconnect_delay
    last_capture_time = 0

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
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            print(f"[INFO] Webcam terhubung. Foto disimpan setiap {CAPTURE_INTERVAL_SEC} detik di: {STORAGE_DIR}")

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cap = None
            print("[WARN] Frame gagal, reconnect...")
            time.sleep(current_delay)
            continue

        now = time.time()
        if now - last_capture_time < CAPTURE_INTERVAL_SEC:
            time.sleep(0.1)
            continue

        last_capture_time = now
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_base = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Simpan foto ke storage
        path_abs, path_rel = save_frame(frame, filename_base)
        if path_abs is None:
            continue
        print(f"[INFO] Foto disimpan: {path_rel}")

        # 2. Deteksi kendaraan pada foto (tipe + warna)
        try:
            det = ensure_detector()
            detections = det.process_frame(frame, process_vehicle_only=True)
        except Exception as e:
            print(f"[ERROR] Deteksi error: {e}")
            continue

        if not detections:
            print(f"[INFO] {path_rel} -> Tidak ada kendaraan")
            if DELETE_IMAGE_IF_NO_VEHICLE:
                if delete_image(path_abs):
                    print(f"[INFO] Gambar dihapus (tidak ada kendaraan): {path_rel}")
            continue

        # 3. Ada kendaraan: log + update JSON
        try:
            log_frame_analysis(path_rel, detections, timestamp_str)
            update_latest_analysis(path_rel, detections, timestamp_str)
            print(f"[INFO] Update: frame_analysis.jsonl + latest_analysis.json")
        except Exception as e:
            print(f"[WARN] Gagal log: {e}")

        for d in detections:
            print(
                f"[INFO] Foto {path_rel} | Kendaraan: {d['vehicle_type']} | "
                f"Warna: {d['color']} | Conf: {d['confidence']:.2f}"
            )
            if LARAVEL_MODE and (now - _last_sent_time >= COOLDOWN_SEC):
                send_to_laravel(d["vehicle_type"], d["color"], d["confidence"])
                _last_sent_time = now
                break


def run_capture_background():
    """Jalankan capture + deteksi di background thread."""
    global _running
    _running = True
    t = threading.Thread(target=capture_and_detect, daemon=True)
    t.start()
    print("[INFO] Background capture+detection started.")


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "running"})


@app.route("/latest", methods=["GET"])
def latest():
    """Return isi latest_analysis.json."""
    try:
        if os.path.exists(LATEST_ANALYSIS_PATH):
            with open(LATEST_ANALYSIS_PATH, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({
            "photo": None,
            "timestamp": None,
            "updated_at": None,
            "detections": [],
            "message": "Belum ada deteksi",
        })
    except Exception as e:
        return jsonify({"error": str(e), "detections": []}), 500


if __name__ == "__main__":
    if os.environ.get("WEBCAM_URL"):
        WEBCAM_URL = os.environ.get("WEBCAM_URL")
    if os.environ.get("LARAVEL_URL"):
        LARAVEL_URL = os.environ.get("LARAVEL_URL")
    if os.environ.get("LARAVEL_MODE", "").lower() in ("1", "true", "yes"):
        LARAVEL_MODE = True

    ensure_storage()
    ensure_detector()

    if LARAVEL_MODE:
        run_capture_background()
        print("[INFO] Mode Laravel. Flask + kirim ke Laravel.")
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    else:
        print("[INFO] Mode CLI: foto tiap 5 detik -> storage -> deteksi (Ctrl+C stop).")
        _running = True
        capture_and_detect()
