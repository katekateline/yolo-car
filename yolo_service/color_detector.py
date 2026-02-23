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
    """Mapping RGB rata-rata ke nama warna (cepat, tanpa K-Means)."""
    v = max(r, g, b)
    v_min = min(r, g, b)
    s = (v - v_min) / (v + 1e-6)
    # Hitam
    if v < 70:
        return "hitam"
    # Putih
    if v > 200 and s < 0.2:
        return "putih"
    # Abu / Silver (saturasi rendah)
    if s < 0.25:
        return "silver" if v > 160 else "abu"
    # Chromatic: dominan hue
    if r >= g and r >= b:
        if g < 100 and b < 100:
            return "merah"
        if r - g < 30 and r - b < 30:
            return "orange"
        return "merah"
    if g >= r and g >= b:
        if r < 80 and b < 80:
            return "hijau"
        if g - r < 40 and g - b < 40:
            return "kuning"
        return "hijau"
    if b >= r and b >= g:
        if r > 150 and g > 100:
            return "ungu"
        return "biru"
    # Coklat: merah+kuning gelap
    if 80 < r < 180 and 50 < g < 150 and b < 100:
        return "coklat"
    return "abu"
