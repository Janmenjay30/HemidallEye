"""
Anti-Spoof Module — Detect photos, screens, and flat surfaces.

Techniques (lightweight, no extra model needed):
  1. Laplacian variance — real faces have texture; photos/screens are smoother
  2. Aspect ratio sanity — reject extreme shapes that aren't real faces
  3. Edge density — real faces have more micro-edges than printed photos

All checks are CPU-only and add < 1 ms per face.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import CFG

logger = logging.getLogger("heimdall.anti_spoof")


class AntiSpoofResult:
    """Result of liveness checks."""

    __slots__ = ("is_live", "laplacian_var", "aspect_ratio", "reasons")

    def __init__(self) -> None:
        self.is_live: bool = True
        self.laplacian_var: float = 0.0
        self.aspect_ratio: float = 1.0
        self.reasons: list[str] = []

    def __repr__(self) -> str:
        return (
            f"AntiSpoofResult(live={self.is_live}, "
            f"lap_var={self.laplacian_var:.1f}, "
            f"ar={self.aspect_ratio:.2f}, "
            f"reasons={self.reasons})"
        )


class AntiSpoof:
    """
    Lightweight anti-spoof checks (no DNN model required).

    These heuristics are designed to catch the most common attack:
    holding up a photo or showing a face on a phone screen.
    They will NOT stop advanced 3D mask attacks — for that, you'd need
    a dedicated anti-spoof model (adds ~200 MB VRAM).
    """

    @staticmethod
    def check(face_crop: np.ndarray, bbox: tuple[int, int, int, int]) -> AntiSpoofResult:
        """
        Run all liveness checks on an aligned or raw face crop.

        Parameters
        ----------
        face_crop : np.ndarray
            BGR face image (any size).
        bbox : tuple
            (x1, y1, x2, y2) of the face in the source image.

        Returns
        -------
        AntiSpoofResult
        """
        result = AntiSpoofResult()

        # ── 1. Laplacian variance (blur / texture check) ─────────────
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        result.laplacian_var = float(lap_var)

        if lap_var < CFG.laplacian_blur_threshold:
            result.is_live = False
            result.reasons.append(
                f"Low texture (Laplacian var={lap_var:.1f} < {CFG.laplacian_blur_threshold})"
            )

        # ── 2. Aspect ratio sanity ───────────────────────────────────
        x1, y1, x2, y2 = bbox
        w, h = max(x2 - x1, 1), max(y2 - y1, 1)
        ar = w / h
        result.aspect_ratio = ar

        if ar < CFG.min_face_aspect_ratio or ar > CFG.max_face_aspect_ratio:
            result.is_live = False
            result.reasons.append(
                f"Abnormal aspect ratio ({ar:.2f}); "
                f"expected [{CFG.min_face_aspect_ratio}, {CFG.max_face_aspect_ratio}]"
            )

        # ── 3. Edge density (Canny) ──────────────────────────────────
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / max(edges.size, 1)

        if edge_density < 0.02:
            result.is_live = False
            result.reasons.append(
                f"Low edge density ({edge_density:.3f} < 0.02); "
                f"possible flat surface"
            )

        if not result.is_live:
            logger.debug("Anti-spoof REJECT: %s", result.reasons)

        return result
