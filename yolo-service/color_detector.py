"""
Deteksi warna dominan kendaraan - metode cepat.
Resize 50x50, rata-rata RGB, mapping ke nama warna.
"""
import cv2
import numpy as np


# Mapping class ID COCO ke tipe kendaraan (sesuai spec)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Nama warna yang didukung
COLOR_NAMES = [
    "merah", "biru", "hitam", "putih", "abu", "silver",
    "hijau", "kuning", "orange", "ungu", "coklat"
]


def get_dominant_color(frame, bbox):
    """
    Ambil warna dominan dari crop bounding box kendaraan.
    Metode cepat: resize kecil (50x50), rata-rata RGB, mapping ke nama warna.

    Args:
        frame: Frame BGR dari OpenCV
        bbox: [x1, y1, x2, y2]

    Returns:
        str: Nama warna (merah, biru, hitam, putih, abu, dll)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    # Clamp ke batas frame
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return "abu"

    vehicle_crop = frame[y1:y2, x1:x2]
    if vehicle_crop.size == 0:
        return "abu"

    # Resize kecil untuk proses cepat
    small = cv2.resize(vehicle_crop, (50, 50))
    # Rata-rata RGB (abaikan pixel terlalu gelap/terang untuk stabil)
    b, g, r = cv2.split(small)
    b_flat = b.flatten()
    g_flat = g.flatten()
    r_flat = r.flatten()
    v_flat = np.maximum(np.maximum(r_flat, g_flat), b_flat)
    valid = (v_flat >= 30) & (v_flat <= 250)
    if valid.sum() < 100:
        valid = v_flat >= 20
    if valid.sum() == 0:
        return "abu"
    r_avg = int(np.mean(r_flat[valid]))
    g_avg = int(np.mean(g_flat[valid]))
    b_avg = int(np.mean(b_flat[valid]))

    return _rgb_to_color_name(r_avg, g_avg, b_avg)


def _rgb_to_color_name(r, g, b):
    """
    Mapping warna rata-rata ke nama warna menggunakan HSV
    (lebih stabil untuk putih/abu/orange/kuning).
    """
    # OpenCV pakai BGR, jadi buat pixel BGR dan konversi ke HSV
    bgr_pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # 1. Hitam / sangat gelap
    if v < 40:
        return "hitam"

    # 2. Putih (sangat terang, saturasi rendah)
    if v > 210 and s < 40:
        return "putih"

    # 3. Abu / silver (saturasi rendah, tapi tidak seputih putih)
    if s < 40:
        if v >= 180:
            return "silver"
        return "abu"

    # 4. Warna kromatik berdasarkan Hue (H: 0–179)
    # Red: sekitar 0 atau 180
    if h < 10 or h >= 170:
        return "merah"
    # Orange: 10–24
    if 10 <= h < 24:
        return "orange"
    # Kuning: 24–35
    if 24 <= h < 35:
        return "kuning"
    # Hijau: 35–85
    if 35 <= h < 85:
        return "hijau"
    # Biru: 85–130
    if 85 <= h < 130:
        return "biru"
    # Ungu / magenta: 130–170
    if 130 <= h < 170:
        return "ungu"

    # Fallback
    return "abu"
