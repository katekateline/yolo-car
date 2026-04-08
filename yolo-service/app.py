"""
YOLO Service: utama = terima gambar dari Laravel (POST /analyze), analisis, balikan JSON;
opsional: IP Webcam (ENABLE_WEBCAM=1) untuk loop capture lama.
"""
import threading
import time
import json
import os
import uuid

import cv2
import numpy as np
import requests
from datetime import datetime
from flask import Flask, jsonify, request

from detector import VehicleDetector
from model_utils import resolve_model_path

app = Flask(__name__)

# Konfigurasi
WEBCAM_URL = "http://localhost:8080/video"
LARAVEL_URL = os.getenv("LARAVEL_URL", "http://localhost:8000/api/detection")

# PlateRecognizer (API plat nomor)
PLATERECOGNIZER_URL = "https://api.platerecognizer.com/v1/plate-reader/"
PLATERECOGNIZER_TOKEN = os.getenv("PLATERECOGNIZER_TOKEN", "")
# Optional: region plate, contoh: ["id"] atau ["us-ca"]
PLATERECOGNIZER_REGIONS = None  # bisa dioverride via env jika perlu

# Mode IP Webcam: kirim POST ke Laravel saat ada kendaraan (cooldown)
LARAVEL_MODE = os.getenv("LARAVEL_MODE", "false").lower() in ("1", "true", "yes")

# Setelah POST /analyze sukses, POST juga ke LARAVEL_URL (opsional; biasanya cukup baca response JSON)
LARAVEL_CALLBACK_AFTER_ANALYZE = os.getenv("LARAVEL_CALLBACK_AFTER_ANALYZE", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Legacy: loop IP Webcam (mati secara default; pakai gambar dari Laravel)
ENABLE_WEBCAM = os.getenv("ENABLE_WEBCAM", "false").lower() in ("1", "true", "yes")

# Opsional: kunci sederhana untuk POST /analyze (header X-API-Key)
YOLO_SERVICE_API_KEY = os.getenv("YOLO_SERVICE_API_KEY", "")

# Ambil foto dari webcam setiap N detik, simpan di folder storage
CAPTURE_INTERVAL_SEC = 5
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# True = hapus foto jika tidak ada kendaraan terdeteksi; False = simpan semua foto
DELETE_IMAGE_IF_NO_VEHICLE = True

# Model YOLO — path di-resolve lewat model_utils (unduhan otomatis jika pakai nama yolov8m.pt)
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


def detect_qr_codes(frame):
    """
    Deteksi QR code dari frame BGR.
    Returns list of dict: [{"text": "...", "points": [[x,y], ...]}]
    """
    detector = cv2.QRCodeDetector()
    results = []
    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(frame)
        if ok and decoded_info is not None:
            for i, text in enumerate(decoded_info):
                clean_text = (text or "").strip()
                if not clean_text:
                    continue
                item = {"text": clean_text}
                if points is not None and len(points) > i and points[i] is not None:
                    pts = points[i].astype(int).tolist()
                    item["points"] = pts
                results.append(item)
            return results
    except Exception:
        pass

    # Fallback satu QR
    try:
        text, pts, _ = detector.detectAndDecode(frame)
        clean_text = (text or "").strip()
        if clean_text:
            item = {"text": clean_text}
            if pts is not None:
                item["points"] = pts.astype(int).tolist()
            results.append(item)
    except Exception:
        return []
    return results


def ensure_storage():
    """Buat folder storage jika belum ada."""
    os.makedirs(STORAGE_DIR, exist_ok=True)


def ensure_detector():
    global _detector
    if _detector is None:
        mp = resolve_model_path()
        if not os.path.isfile(mp):
            print(
                f"[INFO] File model belum ada di disk: {mp} — Ultralytics akan mengunduh otomatis "
                "(butuh koneksi internet pertama kali)."
            )
        _detector = VehicleDetector(model_path=mp, confidence=CONFIDENCE_THRESHOLD)
        print(f"[INFO] Model: {mp}")
    return _detector


def recognize_plate(image_path):
    """
    Panggil API PlateRecognizer untuk membaca plat nomor dari gambar.
    Returns dict minimal: {"plate": str | None} atau None jika gagal/tidak ada.
    """
    if not PLATERECOGNIZER_TOKEN:
        # Jika belum ada token, skip quietly
        return None

    if not os.path.isfile(image_path):
        return None

    try:
        data = {}
        if PLATERECOGNIZER_REGIONS:
            data["regions"] = PLATERECOGNIZER_REGIONS

        with open(image_path, "rb") as fp:
            response = requests.post(
                PLATERECOGNIZER_URL,
                data=data,
                files={"upload": fp},
                headers={"Authorization": f"Token {PLATERECOGNIZER_TOKEN}"},
                timeout=10,
            )

        if not response.ok:
            print(f"[WARN] PlateRecognizer status {response.status_code}: {response.text[:200]}")
            return None

        res_json = response.json()
        results = res_json.get("results") or []
        if not results:
            return None

        first = results[0]
        plate = first.get("plate")
        return {"plate": plate}
    except Exception as e:
        print(f"[WARN] Gagal panggil PlateRecognizer: {e}")
        return None


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


def log_frame_analysis(photo_path, detections, timestamp_str=None, plate_number=None, qr_codes=None):
    """Append hasil analisis ke frame_analysis.jsonl."""
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "photo": photo_path,
        "timestamp": timestamp_str,
        "detections": detections,
        "plate_number": plate_number,
        "qr_codes": qr_codes or [],
    }
    try:
        with open(ANALYSIS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Gagal tulis log: {e}")


def update_latest_analysis(photo_path, detections, timestamp_str=None, plate_number=None, qr_codes=None):
    """Overwrite latest_analysis.json dengan hasil terbaru."""
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "photo": photo_path,
        "timestamp": timestamp_str,
        "updated_at": datetime.now().isoformat(),
        "detections": detections,
        "plate_number": plate_number,
        "qr_codes": qr_codes or [],
    }
    try:
        with open(LATEST_ANALYSIS_PATH, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Gagal update latest: {e}")


def send_to_laravel(vehicle_type, color, confidence, plate_number=None, qr_codes=None):
    """POST satu kendaraan ke Laravel (mode IP Webcam / cooldown)."""
    global _last_sent_time
    data = {
        "vehicle_type": vehicle_type,
        "color": color,
        "confidence": round(confidence, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if plate_number:
        data["plate_number"] = plate_number
    data["qr_codes"] = qr_codes or []
    data["qr_texts"] = [q.get("text") for q in (qr_codes or []) if q.get("text")]
    try:
        r = requests.post(LARAVEL_URL, json=data, timeout=3)
        if r.ok:
            print("[INFO] Data dikirim ke Laravel")
        else:
            print(f"[ERROR] Laravel status {r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gagal kirim: {e}")


def send_analysis_payload_to_laravel(payload):
    """POST hasil lengkap POST /analyze ke Laravel (callback server-to-server)."""
    try:
        r = requests.post(LARAVEL_URL, json=payload, timeout=15)
        if r.ok:
            print("[INFO] Callback hasil analisis ke Laravel OK")
        else:
            print(f"[ERROR] Laravel callback status {r.status_code}: {r.text[:300]}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gagal callback ke Laravel: {e}")


def _check_analyze_api_key():
    if not YOLO_SERVICE_API_KEY:
        return True
    return request.headers.get("X-API-Key") == YOLO_SERVICE_API_KEY


@app.after_request
def _cors_headers(response):
    origin = os.getenv("CORS_ALLOW_ORIGIN", "*")
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/analyze", methods=["OPTIONS"])
def analyze_options():
    return "", 204


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Terima gambar dari Laravel (multipart field `image`), jalankan deteksi + opsional plat.
    Response JSON untuk dipakai langsung oleh Laravel; opsional callback POST ke LARAVEL_URL.
    """
    if not _check_analyze_api_key():
        return jsonify({"success": False, "error": "unauthorized"}), 401

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "missing_file",
            "message": "Kirim multipart dengan field 'image' (file foto).",
        }), 400

    upload = request.files["image"]
    if not upload or upload.filename == "":
        return jsonify({"success": False, "error": "empty_file"}), 400

    raw = upload.read()
    nparr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({
            "success": False,
            "error": "invalid_image",
            "message": "Gambar tidak bisa dibaca (pastikan JPG/PNG).",
        }), 400

    filename_base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    path_abs, path_rel = save_frame(frame, filename_base)
    if path_abs is None:
        return jsonify({"success": False, "error": "save_failed"}), 500

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        det = ensure_detector()
        detections = det.process_frame(frame, process_vehicle_only=True)
    except Exception as e:
        delete_image(path_abs)
        return jsonify({"success": False, "error": "detection_failed", "message": str(e)}), 500

    qr_codes = detect_qr_codes(frame)
    plate_number = None
    if detections:
        plate_info = recognize_plate(path_abs)
        plate_number = plate_info.get("plate") if plate_info else None
        try:
            log_frame_analysis(
                path_rel,
                detections,
                timestamp_str,
                plate_number=plate_number,
                qr_codes=qr_codes,
            )
            update_latest_analysis(
                path_rel,
                detections,
                timestamp_str,
                plate_number=plate_number,
                qr_codes=qr_codes,
            )
        except Exception as e:
            print(f"[WARN] Gagal log: {e}")
    else:
        if DELETE_IMAGE_IF_NO_VEHICLE:
            delete_image(path_abs)
            path_rel = None

    response_body = {
        "success": True,
        "source": "yolo_analyze_upload",
        "timestamp": timestamp_str,
        "updated_at": datetime.now().isoformat(),
        "photo": path_rel,
        "detections": detections,
        "plate_number": plate_number,
        "qr_codes": qr_codes,
        "qr_texts": [q.get("text") for q in qr_codes if q.get("text")],
    }
    if detections:
        first = detections[0]
        response_body["vehicle_type"] = first["vehicle_type"]
        response_body["color"] = first["color"]
        c = first["confidence"]
        response_body["confidence"] = round(float(c), 2) if c is not None else None
    else:
        response_body["vehicle_type"] = None
        response_body["color"] = None
        response_body["confidence"] = None
        response_body["message"] = "no_vehicle"

    if LARAVEL_CALLBACK_AFTER_ANALYZE:
        send_analysis_payload_to_laravel(dict(response_body))

    return jsonify(response_body)


def capture_and_detect():
    """
    Loop utama:
    1. Baca frame dari IP Webcam
    2. Setiap CAPTURE_INTERVAL_SEC detik: simpan foto ke storage
    3. Deteksi kendaraan pada foto itu (ada/tidak, tipe, warna)
    4. Jika ada: log + update JSON (+ kirim Laravel jika mode Laravel)
    """
    global _running, _last_sent_time
    current_delay = _reconnect_delay
    next_capture_time = time.time()
    cap = None

    while _running:
        now = time.time()
        # Tunggu sampai tepat waktu capture berikutnya
        if now < next_capture_time:
            time.sleep(min(0.1, next_capture_time - now))
            continue
        # Atur jadwal capture berikutnya supaya konsisten setiap CAPTURE_INTERVAL_SEC
        next_capture_time += CAPTURE_INTERVAL_SEC

        # Pastikan koneksi ke webcam hanya dibuka sekali dan dipertahankan
        if cap is None or not cap.isOpened():
            print(f"[INFO] Menghubungkan ke webcam: {WEBCAM_URL}")
            cap = cv2.VideoCapture(WEBCAM_URL)
            if not cap.isOpened():
                print(f"[ERROR] Gagal buka stream, retry dalam {current_delay}s...")
                time.sleep(current_delay)
                current_delay = min(current_delay + 2, _max_reconnect_delay)
                cap = None
                continue

            current_delay = _reconnect_delay
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        # Flush beberapa frame supaya tidak pakai frame yang terlalu lama
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Frame gagal, akan coba reconnect...")
            cap.release()
            cap = None
            time.sleep(current_delay)
            continue
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

        # 3. Coba baca QR code + plat nomor via PlateRecognizer
        qr_codes = detect_qr_codes(frame)
        plate_info = recognize_plate(path_abs)
        plate_number = plate_info.get("plate") if plate_info else None

        # 4. Ada kendaraan: log + update JSON (simpan plate_number dan qr_codes)
        try:
            log_frame_analysis(
                path_rel,
                detections,
                timestamp_str,
                plate_number=plate_number,
                qr_codes=qr_codes,
            )
            update_latest_analysis(
                path_rel,
                detections,
                timestamp_str,
                plate_number=plate_number,
                qr_codes=qr_codes,
            )
            print(
                f"[INFO] Update: frame_analysis.jsonl + latest_analysis.json "
                f"(plate={plate_number}, qr={len(qr_codes)})"
            )
        except Exception as e:
            print(f"[WARN] Gagal log: {e}")

        for d in detections:
            print(
                f"[INFO] Foto {path_rel} | Kendaraan: {d['vehicle_type']} | "
                f"Warna: {d['color']} | Conf: {d['confidence']:.2f}"
            )
            if LARAVEL_MODE and (now - _last_sent_time >= COOLDOWN_SEC):
                send_to_laravel(
                    d["vehicle_type"],
                    d["color"],
                    d["confidence"],
                    plate_number=plate_number,
                    qr_codes=qr_codes,
                )
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
            "qr_codes": [],
            "qr_texts": [],
            "plate_number": None,
            "message": "Belum ada deteksi",
        })
    except Exception as e:
        return jsonify({"error": str(e), "detections": []}), 500


if __name__ == "__main__":
    if os.environ.get("WEBCAM_URL"):
        WEBCAM_URL = os.environ.get("WEBCAM_URL")
    if os.environ.get("LARAVEL_URL"):
        LARAVEL_URL = os.environ.get("LARAVEL_URL")

    ensure_storage()
    ensure_detector()

    # Mode lama: hanya loop capture dari IP Webcam tanpa Flask (Ctrl+C stop)
    if os.environ.get("CLI_CAPTURE_ONLY", "").lower() in ("1", "true", "yes"):
        print("[INFO] CLI_CAPTURE_ONLY=1: loop IP Webcam tanpa Flask (Ctrl+C stop).")
        _running = True
        capture_and_detect()
    else:
        if ENABLE_WEBCAM:
            run_capture_background()
            print("[INFO] ENABLE_WEBCAM=1: loop IP Webcam + Flask.")
        else:
            print("[INFO] Mode utama: Laravel kirim gambar ke POST /analyze (ENABLE_WEBCAM=0).")
        port = int(os.environ.get("PORT", "5000"))
        print(f"[INFO] Flask: GET /status | GET /latest | POST /analyze — port {port}")
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
