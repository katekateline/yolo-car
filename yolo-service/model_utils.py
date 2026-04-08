"""
Resolusi path model YOLO: file lokal atau nama standar Ultralytics (unduh otomatis).
"""
import os

# Nama file di root repo (sejajar dengan folder yolo-service)
DEFAULT_ROOT_MODEL = "yolov8m.pt"
# Nama file custom di dalam yolo-service
LOCAL_CUSTOM_MODEL = "best.pt"


def resolve_model_path():
    """
    Return path string yang bisa dipakai VehicleDetector / YOLO().
    Prioritas: yolov8m.pt (root repo) -> best.pt (yolo-service) -> YOLO_MODEL env / yolov8m.pt (unduhan otomatis Ultralytics).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolo_dir = os.path.dirname(os.path.abspath(__file__))

    root_model = os.path.join(repo_root, DEFAULT_ROOT_MODEL)
    if os.path.isfile(root_model):
        return root_model

    local_best = os.path.join(yolo_dir, LOCAL_CUSTOM_MODEL)
    if os.path.isfile(local_best):
        return local_best

    return os.getenv("YOLO_MODEL", DEFAULT_ROOT_MODEL)
