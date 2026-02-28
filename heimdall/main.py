"""
Heimdall — Main Entry Point.

Connects to an RTSP stream (or webcam/video file) and runs the
full Family vs. Stranger recognition pipeline.

Usage:
    python -m heimdall.main --source rtsp://192.168.1.100:554/stream
    python -m heimdall.main --source 0                  # webcam
    python -m heimdall.main --source ./test_video.mp4   # file
    python -m heimdall.main --source rtsp://... --headless  # no GUI
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import cv2
import numpy as np

from .config import CFG
from .pipeline import HeimdallPipeline

logger = logging.getLogger("heimdall")

# ──────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────────────────────────────
_shutdown_requested = False


def _signal_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown signal received — cleaning up...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ──────────────────────────────────────────────────────────────────────
# Overlay drawing
# ──────────────────────────────────────────────────────────────────────
def draw_overlay(
    frame: np.ndarray,
    result,
    pipeline: HeimdallPipeline,
) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    from .decision_engine import DecisionResult

    display = frame.copy()

    # Draw track info
    for track in pipeline._tracker.active_tracks:
        x1, y1, x2, y2 = track.bbox
        cat = track.last_category

        # Color by category
        if cat == "FAMILY":
            color = (0, 200, 0)      # Green
        elif cat == "STRANGER":
            color = (0, 0, 255)      # Red
        elif cat == "UNCERTAIN":
            color = (0, 165, 255)    # Orange
        else:
            color = (200, 200, 200)  # Gray (no decision yet)

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Label
        label_parts = []
        if track.last_name:
            label_parts.append(track.last_name)
        if track.last_similarity > 0:
            label_parts.append(f"{track.last_similarity:.2f}")
        if track.stranger_streak > 0:
            label_parts.append(f"streak:{track.stranger_streak}")

        label = " | ".join(label_parts) if label_parts else f"T#{track.track_id}"

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            display, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    # HUD
    if result:
        hud = (
            f"FPS: {result.fps:.1f} | "
            f"Persons: {result.persons_detected} | "
            f"Faces: {result.faces_detected} | "
            f"Frame: {result.frame_id}"
        )
    else:
        hud = "Skipped frame"

    cv2.putText(
        display, hud, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )

    return display


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────
def run(source: str, headless: bool = False) -> None:
    """Main processing loop."""
    pipeline = HeimdallPipeline()
    pipeline.initialize()

    # Open video source
    try:
        src = int(source)
    except ValueError:
        src = source

    logger.info("Opening video source: %s", src)

    # Use DirectShow backend for webcam on Windows (more reliable)
    if isinstance(src, int):
        import platform
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        logger.error("Failed to open video source: %s", source)
        pipeline.shutdown()
        return

    # RTSP optimizations
    if isinstance(src, str) and src.startswith("rtsp"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # minimize latency
        # Use FFMPEG hardware decoding if available
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"H264"))

    logger.info("Stream opened. Press 'q' to quit.")

    frame_times: list[float] = []
    last_log_time = time.time()

    try:
        while not _shutdown_requested:
            ret, frame = cap.read()
            if not ret:
                # Attempt reconnect for RTSP
                logger.warning("Frame grab failed — attempting reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    logger.error("Reconnect failed. Exiting.")
                    break
                continue

            # Process frame
            result = pipeline.process_frame(frame)

            # Log decisions
            if result and result.decisions:
                for d in result.decisions:
                    if d.should_alert:
                        logger.warning("🚨 %s", d.alert_message)
                    elif d.category == "FAMILY":
                        logger.debug(
                            "✅ %s (sim=%.3f, track=%d)",
                            d.name, d.similarity, d.track_id,
                        )

            # Display
            if not headless:
                display = draw_overlay(frame, result, pipeline)
                cv2.imshow("Heimdall", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # Periodic console stats
            if result:
                frame_times.append(result.processing_time_ms)
            now = time.time()
            if now - last_log_time > 10.0:
                if frame_times:
                    avg = sum(frame_times) / len(frame_times)
                    logger.info(
                        "Stats — avg: %.1f ms/frame (%.1f FPS), "
                        "active tracks: %d",
                        avg,
                        1000.0 / max(avg, 1),
                        len(pipeline._tracker.active_tracks),
                    )
                    frame_times.clear()
                last_log_time = now

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        pipeline.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heimdall — Family vs. Stranger CCTV Recognition",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="RTSP URL, webcam index (0), or video file path",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI display",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run(args.source, args.headless)


if __name__ == "__main__":
    main()
