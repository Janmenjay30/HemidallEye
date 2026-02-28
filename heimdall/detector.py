"""
Person Detector — YOLOv11-nano via ONNX Runtime.

Responsibilities:
  • Preprocess frame → 640×640 FP16 tensor
  • Run inference → raw detections
  • Post-process: filter class==person, apply NMS, reject small boxes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from .config import CFG, YOLO_MODEL_PATH
from .vram_manager import create_onnx_session, session_uses_cuda


def _ort_type_to_numpy(ort_type: str) -> np.dtype:
    """Convert ONNX Runtime type string to numpy dtype."""
    mapping = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}
    return mapping.get(ort_type, np.float32)

logger = logging.getLogger("heimdall.detector")


@dataclass
class PersonDetection:
    """A single person detection result."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in original frame coords
    confidence: float


class PersonDetector:
    """
    YOLOv11-nano person detector running on ONNX Runtime GPU.

    VRAM budget: ~1200 MB (FP16, 640×640 input).
    Latency target: ≤ 8 ms per frame on GTX 1650.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or str(YOLO_MODEL_PATH)
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._input_shape: tuple[int, ...] = ()
        self._input_dtype: np.dtype = np.float32

    # ── Lifecycle ─────────────────────────────────────────────────────
    def load(self) -> None:
        """Load the ONNX model onto the GPU."""
        logger.info("Loading YOLOv11n from %s", self._model_path)
        self._session = create_onnx_session(
            self._model_path,
            gpu_mem_limit_mb=1400,  # generous for YOLO
        )
        meta = self._session.get_inputs()[0]
        self._input_name = meta.name
        self._input_shape = tuple(meta.shape)  # e.g. [1, 3, 640, 640]
        self._input_dtype = _ort_type_to_numpy(meta.type)
        on_gpu = session_uses_cuda(self._session)
        logger.info(
            "YOLOv11n ready \u2014 input: %s %s  dtype=%s  GPU=%s",
            self._input_name, self._input_shape, self._input_dtype, on_gpu,
        )

    def unload(self) -> None:
        """Release the ONNX session and free VRAM."""
        del self._session
        self._session = None
        logger.info("YOLOv11n unloaded")

    # ── Inference ─────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """
        Detect persons in a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image of shape (H, W, 3).

        Returns
        -------
        list[PersonDetection]
            Filtered person detections in original frame coordinates.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        orig_h, orig_w = frame.shape[:2]
        blob, scale, pad = self._preprocess(frame)

        # Run ONNX inference
        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0]  # shape depends on YOLO export variant

        detections = self._postprocess(raw, orig_w, orig_h, scale, pad)
        return detections

    # ── Pre/Post Processing ───────────────────────────────────────────
    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Letter-box resize + normalize → (1, 3, 640, 640) FP16 tensor.

        Returns (blob, scale, (pad_w, pad_h)).
        """
        target = CFG.yolo_input_size
        h, w = frame.shape[:2]
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        pad_w, pad_h = (target - new_w) // 2, (target - new_h) // 2
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # BGR → RGB, HWC → CHW, normalize 0-1
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        blob = blob.astype(self._input_dtype)

        return blob, scale, (pad_w, pad_h)

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad: tuple[int, int],
    ) -> list[PersonDetection]:
        """
        Parse YOLO output, filter person class, apply NMS.

        Handles both YOLOv8/v11 output formats:
          • (1, 84, N) — transposed format (class scores at dim 1)
          • (1, N, 84) — standard format
        """
        # Normalize to shape (N, 84) where 84 = x,y,w,h + 80 class scores
        if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
            # Transposed format (1, 84, N) → (N, 84)
            preds = raw[0].T
        else:
            preds = raw[0]

        # Columns: cx, cy, w, h, class_scores[0..79]
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        class_scores = preds[:, 4:]

        # Filter by person class
        person_id = CFG.yolo_person_class_id
        person_conf = class_scores[:, person_id]
        mask = person_conf >= CFG.yolo_conf_threshold
        if not np.any(mask):
            return []

        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        confs = person_conf[mask]

        # Convert center-wh → xyxy in letterboxed coords
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Remove padding and un-scale to original image
        pad_w, pad_h = pad
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # OpenCV NMS
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        confs_list = confs.astype(float).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes, confs_list, CFG.yolo_conf_threshold, CFG.yolo_iou_threshold
        )

        detections: list[PersonDetection] = []
        if len(indices) > 0:
            for idx in indices:
                i = int(idx)
                bx1, by1, bw_, bh_ = boxes[i]
                bx2, by2 = bx1 + bw_, by1 + bh_
                area = bw_ * bh_
                if area < CFG.min_person_area_px:
                    continue
                detections.append(
                    PersonDetection(
                        bbox=(int(bx1), int(by1), int(bx2), int(by2)),
                        confidence=confs_list[i],
                    )
                )
        return detections
