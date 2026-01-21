import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import colorsys

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
        
        # Color ranges dalam HSV untuk deteksi warna lebih akurat
        self.color_ranges = {
            'merah': [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
            'putih': [(0, 0, 200), (180, 30, 255)],
            'hitam': [(0, 0, 0), (180, 255, 50)],
            'abu-abu': [(0, 0, 50), (180, 30, 200)],
            'biru': [(100, 100, 100), (130, 255, 255)],
            'kuning': [(20, 100, 100), (30, 255, 255)],
            'hijau': [(40, 100, 100), (80, 255, 255)],
            'orange': [(10, 100, 100), (20, 255, 255)],
            'coklat': [(10, 100, 50), (20, 255, 150)],
            'silver': [(0, 0, 150), (180, 20, 220)]
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
        annotated_frame = frame.copy()
        
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
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                    
                    # Anotasi frame
                    # Bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label dengan background
                    label = f"{vehicle_type.upper()} - {color.upper()} ({conf:.2f})"
                    
                    # Hitung ukuran text
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Background untuk text
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - text_h - 10),
                        (x1 + text_w + 10, y1),
                        (0, 255, 0),
                        -1
                    )
                    
                    # Text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )
        
        return annotated_frame, detections
    
    def process_video(self, video_path, output_path=None, show_live=True):
        """
        Proses video dengan deteksi real-time
        
        Args:
            video_path: Path ke video input (atau 0 untuk webcam)
            output_path: Path untuk save video output (optional)
            show_live: Tampilkan preview real-time
        """
        cap = cv2.VideoCapture(video_path)
        
        # Video writer setup
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame, detections = self.process_frame(frame)
            
            # Info frame
            info_text = f"Frame: {frame_count} | Deteksi: {len(detections)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Print deteksi ke console
            if detections:
                print(f"\nFrame {frame_count}:")
                for i, det in enumerate(detections, 1):
                    print(f"  {i}. {det['type']} {det['color']} - "
                          f"Confidence: {det['confidence']:.2%}")
            
            # Save output
            if output_path:
                out.write(annotated_frame)
            
            # Show live
            if show_live:
                cv2.imshow('Vehicle Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Selesai! Total frame: {frame_count}")
    
    def process_image(self, image_path, output_path=None):
        """
        Proses single image
        
        Args:
            image_path: Path ke gambar input
            output_path: Path untuk save hasil (optional)
        
        Returns:
            tuple: (annotated_image, detections)
        """
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
        
        annotated_image, detections = self.process_frame(image)
        
        # Print hasil
        print(f"\nHasil deteksi dari {image_path}:")
        print(f"Total kendaraan terdeteksi: {len(detections)}")
        
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det['type'].upper()} warna {det['color'].upper()} "
                  f"- Confidence: {det['confidence']:.2%}")
        
        # Save jika diminta
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"\n✓ Hasil disimpan ke: {output_path}")
        
        return annotated_image, detections


    def process_all_images(self, input_folder='images/', output_folder='output/'):
        """
        Proses semua gambar di folder untuk demo
        
        Args:
            input_folder: Folder berisi gambar input
            output_folder: Folder untuk save hasil deteksi
        """
        import os
        from pathlib import Path
        
        # Buat folder output jika belum ada
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Extension gambar yang didukung
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # Cari semua gambar di folder (case-insensitive, tanpa duplikasi)
        image_files = []
        seen_files = set()
        
        for file_path in Path(input_folder).iterdir():
            if file_path.is_file():
                # Cek extension (case-insensitive)
                if file_path.suffix.lower() in image_extensions:
                    # Gunakan nama file lowercase untuk deteksi duplikasi
                    file_key = file_path.name.lower()
                    if file_key not in seen_files:
                        seen_files.add(file_key)
                        image_files.append(file_path)
        
        # Sort berdasarkan nama
        image_files.sort()
        
        if not image_files:
            print(f"❌ Tidak ada gambar ditemukan di folder '{input_folder}'")
            print(f"   Pastikan folder '{input_folder}' ada dan berisi gambar")
            return
        
        print("=" * 70)
        print(f"🚗 DEMO DETEKSI KENDARAAN")
        print("=" * 70)
        print(f"📁 Input folder: {input_folder}")
        print(f"📁 Output folder: {output_folder}")
        print(f"🖼️  Total gambar: {len(image_files)}")
        print("=" * 70)
        
        total_vehicles = 0
        
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_file.name}")
            print("-" * 70)
            
            try:
                # Proses gambar
                image = cv2.imread(str(image_file))
                
                if image is None:
                    print(f"   ⚠️  Gagal membaca: {image_file.name}")
                    continue
                
                annotated_image, detections = self.process_frame(image)
                
                # Statistik
                vehicle_count = len(detections)
                total_vehicles += vehicle_count
                
                print(f"   ✓ Terdeteksi: {vehicle_count} kendaraan")
                
                if detections:
                    # Grup berdasarkan tipe
                    types = [d['type'] for d in detections]
                    colors = [d['color'] for d in detections]
                    
                    type_count = Counter(types)
                    color_count = Counter(colors)
                    
                    print(f"   📊 Tipe: {dict(type_count)}")
                    print(f"   🎨 Warna: {dict(color_count)}")
                    
                    # Detail setiap kendaraan
                    for i, det in enumerate(detections, 1):
                        print(f"      {i}. {det['type'].upper()} {det['color']} "
                              f"({det['confidence']:.1%})")
                
                # Save hasil
                output_path = Path(output_folder) / f"detected_{image_file.name}"
                cv2.imwrite(str(output_path), annotated_image)
                print(f"   💾 Saved: {output_path}")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        # Summary
        print("\n" + "=" * 70)
        print("📈 RINGKASAN")
        print("=" * 70)
        print(f"   Total gambar diproses: {len(image_files)}")
        print(f"   Total kendaraan terdeteksi: {total_vehicles}")
        print(f"   Hasil tersimpan di: {output_folder}")
        print("=" * 70)


# ============= DEMO EXECUTION =============

if __name__ == "__main__":
    import os
    
    print("\n🚀 Memulai Demo Deteksi Kendaraan...")
    print("=" * 70)
    
    # Cek folder images
    if not os.path.exists('images'):
        print("❌ Folder 'images/' tidak ditemukan!")
        print("   Membuat folder 'images/'...")
        os.makedirs('images', exist_ok=True)
        print("   ✓ Folder 'images/' telah dibuat")
        print("\n📝 Instruksi:")
        print("   1. Letakkan gambar kendaraan di folder 'images/'")
        print("   2. Jalankan script ini lagi")
        print("=" * 70)
    else:
        # Initialize detector
        # Pilih model sesuai kebutuhan:
        # - yolov8n.pt (paling cepat, akurasi standar)
        # - yolov8s.pt (cepat, akurasi baik)
        # - yolov8m.pt (balance, recommended untuk demo)
        # - yolov8l.pt (lambat, akurasi tinggi)
        # - yolov8x.pt (paling lambat, akurasi maksimal)
        
        try:
            detector = VehicleDetector(
                model_path='model/yolov8m.pt',  # Ubah sesuai model yang ada
                confidence=0.4  # Turunkan sedikit untuk demo agar lebih banyak deteksi
            )
            
            # Proses semua gambar di folder images/
            detector.process_all_images(
                input_folder='images/',
                output_folder='output/'
            )
            
            print("\n✅ Demo selesai! Cek folder 'output/' untuk melihat hasil.")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("\n💡 Troubleshooting:")
            print("   1. Pastikan model ada di folder 'model/' (contoh: model/yolov8m.pt)")
            print("   2. Download model dengan:")
            print("      wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -P model/")
            print("   3. Atau gunakan model lain yang sudah ada")
            print("=" * 70)