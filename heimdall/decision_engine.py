"""
Decision Engine — Thresholding + Temporal Consistency.

Mathematical framework:

Given probe embedding p and gallery embeddings {g_i}:
  sim(p, g_i) = p · g_i   (cosine similarity, both L2-normed)

Decision boundaries:
  FAMILY:    max_i sim(p, g_i) >= tau_accept   (0.45)
  STRANGER:  max_i sim(p, g_i) <  tau_reject   (0.35)
  UNCERTAIN: tau_reject <= max_i sim(p, g_i) < tau_accept

Temporal consistency for track T_k:
  Alert(T_k) = True  if stranger_streak(T_k) >= N  (N=5)

This prevents single-frame false positives (e.g., bad angle, blur).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CFG
from .face_database import FaceDatabase, MatchResult
from .tracker import Track

logger = logging.getLogger("heimdall.decision")


@dataclass
class DecisionResult:
    """Full decision output for one detected face in one frame."""

    track_id: int
    name: str
    similarity: float
    category: str          # "FAMILY" | "UNCERTAIN" | "STRANGER"
    is_live: bool
    stranger_streak: int
    should_alert: bool     # True only after temporal consistency met
    alert_message: str


class DecisionEngine:
    """
    Applies similarity thresholds and temporal consistency.

    One instance per pipeline; it references the shared FaceDatabase
    and updates Track objects in-place.
    """

    def __init__(self, database: FaceDatabase) -> None:
        self._db = database
        self._last_alert_time: dict[int, float] = {}  # track_id → timestamp

    def decide(
        self,
        track: Track,
        embedding: np.ndarray,
        is_live: bool,
    ) -> DecisionResult:
        """
        Make a family/stranger decision for one face observation.

        Parameters
        ----------
        track : Track
            The tracker's Track object (will be mutated).
        embedding : np.ndarray
            512-D probe embedding.
        is_live : bool
            Result of anti-spoof check.

        Returns
        -------
        DecisionResult
        """
        # ── 1. Query database ────────────────────────────────────────
        match: MatchResult = self._db.query(embedding)

        # ── 2. Update track state ────────────────────────────────────
        track.last_category = match.category
        track.last_name = match.name
        track.last_similarity = match.similarity

        # ── 3. Temporal consistency ──────────────────────────────────
        if match.category == "STRANGER" and is_live:
            track.stranger_streak += 1
        elif match.category == "FAMILY":
            # Reset streak — clearly recognized
            track.stranger_streak = 0
        else:
            # UNCERTAIN or not live — don't increment, but don't reset
            pass

        # ── 4. Alert logic ───────────────────────────────────────────
        should_alert = False
        alert_msg = ""

        if (
            track.stranger_streak >= CFG.stranger_consecutive_frames
            and is_live
            and not track.alert_fired
        ):
            # Check cooldown
            now = time.time()
            last = self._last_alert_time.get(track.track_id, 0.0)
            if now - last >= CFG.alert_cooldown_sec:
                should_alert = True
                track.alert_fired = True
                self._last_alert_time[track.track_id] = now
                alert_msg = (
                    f"⚠ STRANGER ALERT — Track #{track.track_id} "
                    f"(sim={match.similarity:.3f}) seen for "
                    f"{track.stranger_streak} consecutive frames"
                )
                logger.warning(alert_msg)

        if not is_live:
            alert_msg = (
                f"Anti-spoof: face on Track #{track.track_id} "
                f"failed liveness — ignoring"
            )

        return DecisionResult(
            track_id=track.track_id,
            name=match.name,
            similarity=match.similarity,
            category=match.category,
            is_live=is_live,
            stranger_streak=track.stranger_streak,
            should_alert=should_alert,
            alert_message=alert_msg,
        )

    @staticmethod
    def save_stranger_snapshot(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        track_id: int,
    ) -> str | None:
        """Save a cropped image of the stranger for review."""
        if not CFG.log_stranger_snapshots:
            return None

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"stranger_track{track_id}_{ts}.jpg"
        path = Path(CFG.snapshot_dir) / filename
        cv2.imwrite(str(path), crop)
        logger.info("Saved stranger snapshot: %s", path)
        return str(path)
