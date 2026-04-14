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
    Ambil warna dominan dari body kendaraan menggunakan K-Means Clustering.
    Fokus pada area tengah body untuk menghindari gangguan background, ban, dan kaca.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_frame, w_frame = frame.shape[:2]
    
    # Clamp ke batas frame
    x1, x2 = max(0, x1), min(w_frame, x2)
    y1, y2 = max(0, y1), min(h_frame, y2)
    
    if x2 <= x1 or y2 <= y1:
        return "abu"

    # --- CROP AREA BODY (CENTRAL CROP) ---
    # Kita ambil area tengah (40% - 70% tinggi, 20% - 80% lebar) 
    # untuk mendapatkan warna cat body, bukan kaca/ban/background
    w = x2 - x1
    h = y2 - y1
    cx1 = x1 + int(w * 0.2)
    cx2 = x1 + int(w * 0.8)
    cy1 = y1 + int(h * 0.3) # Hindari atap/kaca depan
    cy2 = y1 + int(h * 0.7) # Hindari ban/kolong
    
    body_crop = frame[cy1:cy2, cx1:cx2]
    if body_crop.size == 0:
        body_crop = frame[y1:y2, x1:x2] # Fallback ke full bbox

    # Resize untuk mempercepat K-Means
    small = cv2.resize(body_crop, (40, 40), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    # Filter pixel yang terlalu gelap (hitam/bayangan) atau terlalu terang (refleksi cahaya)
    # agar tidak mengacaukan warna dominan cat
    v_max = np.max(pixels, axis=1)
    v_min = np.min(pixels, axis=1)
    # Masking: value tidak boleh terlalu gelap (<30) dan saturasi tidak boleh nol jika sangat terang
    mask = (v_max > 30) & (v_max < 250)
    filtered_pixels = pixels[mask]

    if len(filtered_pixels) < 10:
        filtered_pixels = pixels # Fallback jika terlalu banyak yang terfilter

    # --- K-MEANS CLUSTERING (K=3) ---
    # Mencari 3 kelompok warna utama (Cat Body, Bayangan, Refleksi)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = min(3, len(filtered_pixels))
    _, labels, centers = cv2.kmeans(filtered_pixels, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    # Hitung jumlah pixel per cluster
    counts = np.bincount(labels.flatten())
    
    # Cari cluster dengan saturasi tertinggi atau yang paling dominan
    # Cat kendaraan biasanya memiliki saturasi lebih tinggi daripada bayangan/abu-abu
    best_color = None
    max_score = -1

    for i in range(len(centers)):
        b, g, r = centers[i]
        # Konversi ke HSV untuk analisa kualitas warna
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
        hue, sat, val = hsv
        
        # Skor: Gabungan antara dominansi jumlah pixel dan saturasi
        # Warna cat asli biasanya dominan DAN punya saturasi
        score = counts[i] * (sat + 1) 
        
        if score > max_score:
            max_score = score
            best_color = (r, g, b)

    if best_color is None:
        return "abu"

    return _rgb_to_color_name(best_color[0], best_color[1], best_color[2])


def _rgb_to_color_name(r, g, b):
    """
    Mapping warna ke nama Indonesia dengan threshold HSV yang lebih presisi.
    """
    bgr_pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # 1. Hitam (Value sangat rendah)
    if v < 45:
        return "hitam"

    # 2. Putih (Saturasi rendah, Value tinggi)
    if s < 35 and v > 190:
        return "putih"

    # 3. Abu-abu / Silver (Saturasi rendah)
    if s < 45:
        if v > 160: return "silver"
        return "abu"

    # 4. Warna Berdasarkan Hue (0-179)
    # Merah (0-8 atau 165-179)
    if h < 8 or h >= 165:
        return "merah"
    
    # Orange (8-20) - Diperketat agar tidak campur dengan merah/kuning
    if 8 <= h < 20:
        return "orange"
    
    # Kuning (20-36)
    if 20 <= h < 36:
        return "kuning"
    
    # Hijau (36-85)
    if 36 <= h < 85:
        return "hijau"
    
    # Biru (85-135)
    if 85 <= h < 135:
        return "biru"
    
    # Ungu / Magenta (135-165)
    if 135 <= h < 165:
        # Tambahan: Cek jika saturasi ungu rendah, bisa jadi itu refleksi kebiruan/abu
        if s < 60: return "abu"
        return "ungu"

    return "abu"
