"""
Detektor YOLO untuk kendaraan: car, motorcycle, truck, bus.
Filter hanya class tersebut, return list deteksi dengan bbox dan class.
"""
from ultralytics import YOLO
import numpy as np

from color_detector import VEHICLE_CLASSES, get_dominant_color


class VehicleDetector:
    def __init__(self, model_path="best.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_class_ids = set(VEHICLE_CLASSES.keys())

    def get_vehicle_type(self, class_id):
        """Return nama tipe: car, motorcycle, truck, bus."""
        return VEHICLE_CLASSES.get(class_id, "car")

    def process_frame(self, frame, process_vehicle_only=True):
        """
        Jalankan YOLO, filter hanya kendaraan, tambah warna.
        Returns list of dict: type, color, confidence, bbox.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            verbose=False,
            imgsz=640,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                if process_vehicle_only and class_id not in self.vehicle_class_ids:
                    continue
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                vehicle_type = self.get_vehicle_type(class_id)
                color = get_dominant_color(frame, bbox)
                detections.append({
                    "vehicle_type": vehicle_type,
                    "color": color,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                })
        return detections
