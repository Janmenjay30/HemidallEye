"""
Main Pipeline Orchestrator — Ties everything together.

Data flow per frame:
  Frame → Person Detection → Face Detection → Recognition → Decision

Optimizations for GTX 1650:
  • Frame skipping (adaptive based on processing time)
  • Person-crop-only face detection (not full frame)
  • Sequential model inference (keeps VRAM stable, no overlap)
  • FP16 throughout
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .anti_spoof import AntiSpoof
from .config import CFG
from .decision_engine import DecisionEngine, DecisionResult
from .detector import PersonDetector
from .face_database import FaceDatabase
from .face_detector import FaceDetector
from .recognizer import FaceRecognizer
from .tracker import IoUTracker, Track
from .vram_manager import get_vram_status

logger = logging.getLogger("heimdall.pipeline")


@dataclass
class FrameResult:
    """Complete processing result for one frame."""

    frame_id: int
    timestamp: float
    persons_detected: int
    faces_detected: int
    decisions: list[DecisionResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    fps: float = 0.0


class HeimdallPipeline:
    """
    The main CCTV processing pipeline.

    Usage:
        pipeline = HeimdallPipeline()
        pipeline.initialize()

        cap = cv2.VideoCapture(rtsp_url)
        while True:
            ret, frame = cap.read()
            result = pipeline.process_frame(frame)
            # Handle result.decisions ...
    """

    def __init__(self, config: type | None = None) -> None:
        # Components (lazy-initialized)
        self._person_detector = PersonDetector()
        self._face_detector = FaceDetector()
        self._recognizer = FaceRecognizer()
        self._database = FaceDatabase()
        self._tracker = IoUTracker()
        self._anti_spoof = AntiSpoof()
        self._decision_engine: DecisionEngine | None = None

        # State
        self._frame_count: int = 0
        self._initialized: bool = False

        # Adaptive frame skip
        self._skip_counter: int = 0
        self._avg_process_ms: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────
    def initialize(self) -> None:
        """Load all models and the face database."""
        logger.info("=" * 60)
        logger.info("HEIMDALL — Initializing Pipeline")
        logger.info("=" * 60)

        vram = get_vram_status()
        logger.info(
            "GPU VRAM: %.0f MB total, %.0f MB free",
            vram["total_mb"],
            vram["free_mb"],
        )

        # Load models sequentially (safest for VRAM)
        logger.info("[1/4] Loading person detector (YOLOv11n)...")
        self._person_detector.load()

        logger.info("[2/4] Loading face detector (SCRFD)...")
        self._face_detector.load()

        logger.info("[3/4] Loading face recognizer (ArcFace)...")
        self._recognizer.load()

        logger.info("[4/4] Loading face database...")
        self._database.load()

        self._decision_engine = DecisionEngine(self._database)
        self._initialized = True

        vram = get_vram_status()
        logger.info(
            "All models loaded — VRAM: %.0f MB used, %.0f MB free",
            vram["used_mb"],
            vram["free_mb"],
        )
        logger.info("Enrolled persons: %d", self._database.get_person_count())
        logger.info("Total embeddings: %d", self._database.get_total_embeddings())
        logger.info("=" * 60)

    def shutdown(self) -> None:
        """Release all resources."""
        logger.info("Shutting down pipeline...")
        self._person_detector.unload()
        self._face_detector.unload()
        self._recognizer.unload()
        if self._database._dirty:
            self._database.save()
        self._tracker.reset()
        self._initialized = False
        logger.info("Pipeline shut down.")

    # ── Main Processing ───────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> FrameResult | None:
        """
        Process a single BGR frame through the full pipeline.

        Returns None if the frame is skipped (adaptive frame skip).
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call .initialize() first.")

        self._frame_count += 1

        # ── Adaptive frame skip ──────────────────────────────────────
        self._skip_counter += 1
        current_skip = self._compute_adaptive_skip()
        if self._skip_counter < current_skip:
            return None
        self._skip_counter = 0

        t_start = time.perf_counter()

        # ── Stage 1: Person detection ────────────────────────────────
        person_detections = self._person_detector.detect(frame)

        # ── Stage 2: Update tracker ──────────────────────────────────
        det_boxes = [d.bbox for d in person_detections]
        tracks = self._tracker.update(det_boxes)

        # Build mapping: track → detection (by IoU)
        track_det_map = self._match_tracks_to_detections(tracks, person_detections)

        # ── Stage 3-5: Face detect → Recognize → Decide ─────────────
        decisions: list[DecisionResult] = []
        faces_detected = 0

        for track in tracks:
            if track.age > 0:
                # Track wasn't matched this frame — skip
                continue

            det = track_det_map.get(track.track_id)
            if det is None:
                continue

            x1, y1, x2, y2 = det.bbox
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Face detection within person crop
            face_results = self._face_detector.detect_faces(person_crop)
            if not face_results:
                continue

            faces_detected += len(face_results)

            # Process the largest / most confident face per person
            best_face = max(face_results, key=lambda f: f.confidence)

            # Anti-spoof check
            spoof_result = self._anti_spoof.check(
                best_face.aligned_face, best_face.bbox
            )

            # Get embedding
            embedding = self._recognizer.get_embedding(best_face.aligned_face)

            # Decision
            decision = self._decision_engine.decide(
                track, embedding, spoof_result.is_live
            )
            decisions.append(decision)

            # Save snapshot if stranger alert
            if decision.should_alert:
                DecisionEngine.save_stranger_snapshot(
                    frame, det.bbox, track.track_id
                )

        # ── Timing ───────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._avg_process_ms = 0.9 * self._avg_process_ms + 0.1 * elapsed_ms
        fps = 1000.0 / max(elapsed_ms, 1)

        result = FrameResult(
            frame_id=self._frame_count,
            timestamp=time.time(),
            persons_detected=len(person_detections),
            faces_detected=faces_detected,
            decisions=decisions,
            processing_time_ms=elapsed_ms,
            fps=fps,
        )

        return result

    # ── Helpers ───────────────────────────────────────────────────────
    def _compute_adaptive_skip(self) -> int:
        """
        Dynamically adjust frame skip based on processing speed.

        Target: CFG.target_fps (15 FPS).
        If processing is slow, skip more frames to maintain responsiveness.
        """
        if self._avg_process_ms <= 0:
            return CFG.frame_skip

        # Desired ms/frame for target FPS
        target_ms = 1000.0 / CFG.target_fps
        if self._avg_process_ms > target_ms:
            # Processing is slower than target — increase skip
            return max(CFG.frame_skip, int(self._avg_process_ms / target_ms))
        return CFG.frame_skip

    @staticmethod
    def _match_tracks_to_detections(tracks, detections):
        """
        Simple mapping of tracks to detections by closest bbox.

        Since the tracker already matched by IoU, we re-match here
        using the same bbox for track→detection association.
        """
        from .detector import PersonDetection

        result = {}
        used = set()

        for track in tracks:
            best_iou = -1.0
            best_det = None
            for i, det in enumerate(detections):
                if i in used:
                    continue
                iou = _compute_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
                    best_idx = i
            if best_det is not None and best_iou > 0.3:
                result[track.track_id] = best_det
                used.add(best_idx)

        return result

    @property
    def database(self) -> FaceDatabase:
        return self._database


def _compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
