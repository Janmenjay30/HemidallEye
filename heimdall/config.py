"""
Heimdall Configuration — All tunable parameters in one place.

Designed for GTX 1650 (4 GB VRAM). Every value has a conservative default
that keeps peak VRAM under 2.6 GB, leaving ~1.4 GB headroom.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
PHOTOS_DIR = PROJECT_ROOT / "photos"

# Ensure directories exist
for _dir in (MODELS_DIR, DATA_DIR, LOGS_DIR, PHOTOS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Model file names (placed inside MODELS_DIR)
# ──────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = MODELS_DIR / "yolo11n.onnx"
SCRFD_MODEL_PATH = MODELS_DIR / "det_10g.onnx"        # SCRFD-10GF from buffalo_l
ARCFACE_MODEL_PATH = MODELS_DIR / "w600k_r50.onnx"


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration."""

    # ── Stream ────────────────────────────────────────────────────────
    rtsp_url: str = "rtsp://192.168.1.100:554/stream"
    target_fps: int = 15                   # minimum acceptable FPS
    frame_skip: int = 2                    # process every Nth frame (adaptive)
    stream_timeout_sec: float = 10.0       # reconnect if no frame for this long

    # ── Detection (YOLOv11-nano) ──────────────────────────────────────
    yolo_input_size: int = 640             # NxN input resolution
    yolo_conf_threshold: float = 0.45      # min detection confidence
    yolo_iou_threshold: float = 0.50       # NMS IoU threshold
    yolo_person_class_id: int = 0          # COCO class 0 = person
    min_person_area_px: int = 3000         # ignore tiny detections

    # ── Face Detection (SCRFD) ────────────────────────────────────────
    scrfd_input_size: int = 640
    scrfd_conf_threshold: float = 0.50
    scrfd_iou_threshold: float = 0.40
    min_face_size_px: int = 40             # ignore faces smaller than 40×40

    # ── Face Recognition (ArcFace) ────────────────────────────────────
    arcface_input_size: int = 112          # 112×112 aligned face
    embedding_dim: int = 512

    # ── Similarity Thresholds ─────────────────────────────────────────
    #   FAMILY:  sim >= tau_accept
    #   GREY:    tau_reject <= sim < tau_accept
    #   STRANGER: sim < tau_reject
    tau_accept: float = 0.45
    tau_reject: float = 0.35

    # ── Temporal Consistency ──────────────────────────────────────────
    stranger_consecutive_frames: int = 5   # N consecutive STRANGER before alert
    track_max_age: int = 30                # frames before a lost track is pruned
    track_iou_threshold: float = 0.30      # IoU to match tracks across frames

    # ── Anti-Spoof ────────────────────────────────────────────────────
    laplacian_blur_threshold: float = 50.0 # variance below this → flat / photo
    min_face_aspect_ratio: float = 0.6     # reject extreme aspect ratios
    max_face_aspect_ratio: float = 1.8

    # ── VRAM Management ───────────────────────────────────────────────
    vram_total_mb: int = 4096              # GTX 1650
    vram_safety_margin_mb: int = 512       # keep this much free
    onnx_arena_mem_limit_mb: int = 256     # per-session arena limit
    use_fp16: bool = True                  # FP16 inference for all models

    # ── ONNX Runtime ──────────────────────────────────────────────────
    onnx_device_id: int = 0               # CUDA device index
    onnx_log_level: int = 3               # WARNING

    # ── Database ──────────────────────────────────────────────────────
    faiss_index_path: str = str(DATA_DIR / "family_db.faiss")
    family_meta_path: str = str(DATA_DIR / "family_meta.json")
    max_gallery_per_person: int = 10       # embeddings stored per identity

    # ── Logging / Alerts ──────────────────────────────────────────────
    log_stranger_snapshots: bool = True
    snapshot_dir: str = str(LOGS_DIR)
    alert_cooldown_sec: float = 30.0       # min seconds between alerts for same track


# Global default config (importable everywhere)
CFG = PipelineConfig()
