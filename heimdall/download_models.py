"""
Model Downloader — Fetches ONNX models from public sources.

Models required:
  1. YOLOv11n (Ultralytics) — exported to ONNX
  2. SCRFD-10GF (InsightFace buffalo_l pack) — face detection
  3. ArcFace w600k_r50 (InsightFace buffalo_l pack) — face recognition

InsightFace models are distributed as a zip archive ("buffalo_l").
We download the zip, extract the two ONNX files we need, and clean up.
"""

from __future__ import annotations

import logging
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from .config import MODELS_DIR, YOLO_MODEL_PATH, SCRFD_MODEL_PATH, ARCFACE_MODEL_PATH

logger = logging.getLogger("heimdall.downloader")

# ──────────────────────────────────────────────────────────────────────
# Model URLs
# ──────────────────────────────────────────────────────────────────────
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"

# buffalo_l contains: det_10g.onnx (SCRFD-10GF) + w600k_r50.onnx (ArcFace)
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# Mapping: filename inside the zip → destination path
BUFFALO_EXTRACT_MAP = {
    "det_10g.onnx": SCRFD_MODEL_PATH,
    "w600k_r50.onnx": ARCFACE_MODEL_PATH,
}


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """Download a file with progress bar. Returns True on success."""
    if dest.exists():
        logger.info("Already exists: %s", dest.name)
        return True

    logger.info("Downloading %s → %s", description or url, dest.name)
    try:
        resp = requests.get(url, stream=True, timeout=120, allow_redirects=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info("Downloaded: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True

    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        if dest.exists():
            dest.unlink()
        return False


def download_and_extract_buffalo() -> bool:
    """
    Download the InsightFace buffalo_l.zip and extract the ONNX models we need.

    Returns True if both det_10g.onnx and w600k_r50.onnx are available.
    """
    # Check if both target files already exist
    all_present = all(p.exists() for p in BUFFALO_EXTRACT_MAP.values())
    if all_present:
        logger.info("InsightFace models already present — skipping download.")
        return True

    zip_path = MODELS_DIR / "buffalo_l.zip"

    # Download the zip
    if not zip_path.exists():
        ok = download_file(BUFFALO_L_URL, zip_path, "InsightFace buffalo_l pack (~300 MB)")
        if not ok:
            return False

    # Extract the specific ONNX files
    logger.info("Extracting models from buffalo_l.zip ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            for zip_name, dest_path in BUFFALO_EXTRACT_MAP.items():
                if dest_path.exists():
                    logger.info("  Already extracted: %s", dest_path.name)
                    continue

                # Find the file inside the zip (may be in a subfolder)
                matches = [n for n in names if n.endswith(zip_name)]
                if not matches:
                    logger.error("  '%s' not found in zip! Contents: %s", zip_name, names)
                    return False

                # Extract to models dir
                source = matches[0]
                data = zf.read(source)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(data)
                logger.info("  Extracted: %s (%.1f MB)", dest_path.name, len(data) / 1e6)

    except zipfile.BadZipFile:
        logger.error("Corrupt zip file — deleting and retry on next run.")
        zip_path.unlink(missing_ok=True)
        return False

    # Clean up zip to save disk space
    zip_path.unlink(missing_ok=True)
    logger.info("Cleaned up buffalo_l.zip")
    return True


def download_all_models() -> bool:
    """Download all required models. Returns True if all succeeded."""
    print(f"\n{'='*60}")
    print("HEIMDALL — Model Downloader")
    print(f"Destination: {MODELS_DIR}")
    print(f"{'='*60}\n")

    all_ok = True

    # 1. YOLO
    ok = download_file(YOLO_URL, YOLO_MODEL_PATH, "YOLOv11-nano (ONNX, ~11 MB)")
    if ok:
        print("  ✓ Ready:  yolo11n")
    else:
        print("  ✗ Failed: yolo11n")
        all_ok = False

    # 2 & 3. InsightFace (SCRFD + ArcFace from buffalo_l)
    ok = download_and_extract_buffalo()
    if ok:
        print("  ✓ Ready:  scrfd (det_10g.onnx)")
        print("  ✓ Ready:  arcface (w600k_r50.onnx)")
    else:
        print("  ✗ Failed: InsightFace models (buffalo_l)")
        all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("All models downloaded successfully!")
    else:
        print("Some models failed to download. Check URLs and retry.")
    print(f"{'='*60}\n")

    return all_ok


# ──────────────────────────────────────────────────────────────────────
# Manual YOLO export instructions (if URL is unavailable)
# ──────────────────────────────────────────────────────────────────────
YOLO_EXPORT_INSTRUCTIONS = """
If the YOLOv11n ONNX download fails, export manually:

    pip install ultralytics
    python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='onnx', imgsz=640, half=True, simplify=True)
"
    # Then move yolo11n.onnx to models/yolo11n.onnx
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = download_all_models()
    if not success:
        print(YOLO_EXPORT_INSTRUCTIONS)
        sys.exit(1)
