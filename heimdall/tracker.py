"""
IoU Tracker — Lightweight multi-object tracker for temporal consistency.

Uses simple IoU-based assignment (no deep features needed).
Each person/face gets a track ID that persists across frames,
enabling the Decision Engine to count consecutive stranger detections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import CFG

logger = logging.getLogger("heimdall.tracker")


@dataclass
class Track:
    """A tracked entity across frames."""

    track_id: int
    bbox: tuple[int, int, int, int]  # latest (x1, y1, x2, y2)
    age: int = 0                     # frames since last update
    hits: int = 1                    # total frames matched

    # Decision engine state
    stranger_streak: int = 0         # consecutive STRANGER classifications
    last_category: str = ""          # last classification result
    last_name: str = ""
    last_similarity: float = 0.0
    alert_fired: bool = False        # True if an alert was already sent


class IoUTracker:
    """
    Frame-to-frame tracker using IoU overlap.

    Algorithm:
      1. Compute IoU matrix between existing tracks and new detections.
      2. Greedy assign (Hungarian is overkill for ≤ 10 targets).
      3. Update matched tracks, create new ones, age unmatched.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1

    @property
    def active_tracks(self) -> list[Track]:
        """Return all currently active tracks."""
        return list(self._tracks.values())

    def update(
        self, detections: list[tuple[int, int, int, int]]
    ) -> list[Track]:
        """
        Update tracker with new frame detections.

        Parameters
        ----------
        detections : list of (x1, y1, x2, y2)
            Person bounding boxes in the current frame.

        Returns
        -------
        list[Track]
            Updated tracks (includes newly created ones).
        """
        if not self._tracks:
            # First frame: create tracks for all detections
            for bbox in detections:
                self._create_track(bbox)
            return self.active_tracks

        if not detections:
            # No detections: age all tracks
            self._age_tracks()
            return self.active_tracks

        # Compute IoU matrix
        track_list = list(self._tracks.values())
        track_boxes = [t.bbox for t in track_list]
        iou_matrix = self._compute_iou_matrix(track_boxes, detections)

        # Greedy matching
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Sort by IoU (highest first)
        pairs = []
        for ti in range(len(track_list)):
            for di in range(len(detections)):
                pairs.append((iou_matrix[ti, di], ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for iou_val, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            if iou_val < CFG.track_iou_threshold:
                break
            # Match!
            track = track_list[ti]
            track.bbox = detections[di]
            track.age = 0
            track.hits += 1
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Create new tracks for unmatched detections
        for di in range(len(detections)):
            if di not in matched_dets:
                self._create_track(detections[di])

        # Age unmatched tracks
        self._age_tracks(exclude_ids={track_list[ti].track_id for ti in matched_tracks})

        return self.active_tracks

    def get_track(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    # ── Private ───────────────────────────────────────────────────────
    def _create_track(self, bbox: tuple[int, int, int, int]) -> Track:
        track = Track(track_id=self._next_id, bbox=bbox)
        self._tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _age_tracks(self, exclude_ids: set[int] | None = None) -> None:
        """Age unmatched tracks and prune stale ones."""
        exclude = exclude_ids or set()
        to_delete = []

        for tid, track in self._tracks.items():
            if tid not in exclude:
                track.age += 1
                if track.age > CFG.track_max_age:
                    to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

    @staticmethod
    def _compute_iou_matrix(
        boxes_a: list[tuple[int, int, int, int]],
        boxes_b: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Compute IoU between two sets of boxes. Returns (len_a, len_b) matrix."""
        a = np.array(boxes_a, dtype=np.float32)
        b = np.array(boxes_b, dtype=np.float32)

        # Intersection
        x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
        y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
        x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
        y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Union
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter

        return np.where(union > 0, inter / union, 0.0)
