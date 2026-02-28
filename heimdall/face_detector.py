"""
Face Detector — SCRFD-2.5GF via ONNX Runtime.

Responsibilities:
  • Run face detection inside a person crop (saves compute vs. full-frame)
  • Return bounding boxes + 5-point landmarks
  • Align face to 112×112 using ArcFace-standard alignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from .config import CFG, SCRFD_MODEL_PATH
from .vram_manager import create_onnx_session, session_uses_cuda


def _ort_type_to_numpy(ort_type: str) -> np.dtype:
    """Convert ONNX Runtime type string to numpy dtype."""
    mapping = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}
    return mapping.get(ort_type, np.float32)

logger = logging.getLogger("heimdall.face_detector")

# ──────────────────────────────────────────────────────────────────────
# ArcFace standard alignment reference points (for 112×112 output)
# ──────────────────────────────────────────────────────────────────────
ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class FaceDetection:
    """A single detected face with landmark info."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in crop coordinates
    confidence: float
    landmarks: np.ndarray            # shape (5, 2) — pixel coords in crop
    aligned_face: np.ndarray         # 112×112×3 BGR aligned face


class FaceAligner:
    """Align a face crop to 112×112 using similarity transform on 5 landmarks."""

    @staticmethod
    def align(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply ArcFace-standard similarity transform.

        Parameters
        ----------
        image : np.ndarray
            Source image (BGR).
        landmarks : np.ndarray
            5-point landmarks, shape (5, 2).

        Returns
        -------
        np.ndarray
            Aligned face, 112×112×3 BGR.
        """
        dst = ARCFACE_REF_LANDMARKS.copy()
        tform = _umeyama(landmarks, dst, estimate_scale=True)
        aligned = cv2.warpAffine(
            image, tform[:2], (112, 112), borderValue=0.0
        )
        return aligned


def _umeyama(
    src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True
) -> np.ndarray:
    """
    Estimate N-D similarity transformation with or without scaling.

    Reimplemented to avoid skimage dependency.
    """
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)

    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T

    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale

    return T


class FaceDetector:
    """
    SCRFD-2.5GF face detector.

    VRAM budget: ~300 MB (FP16).
    Runs on cropped person regions only.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or str(SCRFD_MODEL_PATH)
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._aligner = FaceAligner()
        self._input_dtype: np.dtype = np.float32

    def load(self) -> None:
        """Load SCRFD model onto GPU."""
        logger.info("Loading SCRFD from %s", self._model_path)
        self._session = create_onnx_session(
            self._model_path,
            gpu_mem_limit_mb=400,
        )
        meta = self._session.get_inputs()[0]
        self._input_name = meta.name
        self._input_dtype = _ort_type_to_numpy(meta.type)
        on_gpu = session_uses_cuda(self._session)
        logger.info("SCRFD ready \u2014 input: %s  dtype=%s  GPU=%s", self._input_name, self._input_dtype, on_gpu)

    def unload(self) -> None:
        del self._session
        self._session = None
        logger.info("SCRFD unloaded")

    def detect_faces(
        self, person_crop: np.ndarray
    ) -> list[FaceDetection]:
        """
        Detect faces inside a person crop.

        Parameters
        ----------
        person_crop : np.ndarray
            BGR image of a person (sub-region of full frame).

        Returns
        -------
        list[FaceDetection]
            Detected faces with aligned 112×112 crops.
        """
        if self._session is None:
            raise RuntimeError("SCRFD not loaded. Call .load() first.")

        h, w = person_crop.shape[:2]
        blob, scale = self._preprocess(person_crop)

        outputs = self._session.run(None, {self._input_name: blob})

        faces = self._postprocess(outputs, w, h, scale, person_crop)
        return faces

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Resize to model input and normalize."""
        target = CFG.scrfd_input_size
        h, w = image.shape[:2]
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Pad to target size
        blob = np.zeros((target, target, 3), dtype=np.uint8)
        blob[:new_h, :new_w] = resized

        # Normalize
        blob_float = (blob.astype(np.float32) - 127.5) / 128.0
        blob_float = blob_float.transpose(2, 0, 1)[np.newaxis]

        blob_float = blob_float.astype(self._input_dtype)

        return blob_float, scale

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        orig_w: int,
        orig_h: int,
        scale: float,
        source_image: np.ndarray,
    ) -> list[FaceDetection]:
        """
        Decode SCRFD multi-stride outputs.

        SCRFD outputs scores, bboxes, and keypoints at 3 strides (8, 16, 32).
        Output order: [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
        """
        # Handle different SCRFD output formats
        num_outputs = len(outputs)

        if num_outputs == 9:
            # Standard SCRFD with landmarks: 3 score + 3 bbox + 3 kps
            strides = [8, 16, 32]
            all_boxes = []
            all_scores = []
            all_kps = []

            for idx, stride in enumerate(strides):
                scores = outputs[idx]
                bboxes = outputs[idx + 3]
                kps = outputs[idx + 6]

                self._decode_stride(
                    scores, bboxes, kps, stride, scale,
                    orig_w, orig_h, all_boxes, all_scores, all_kps,
                )

        elif num_outputs == 6:
            # SCRFD without landmarks: 3 score + 3 bbox
            strides = [8, 16, 32]
            all_boxes = []
            all_scores = []
            all_kps = []

            for idx, stride in enumerate(strides):
                scores = outputs[idx]
                bboxes = outputs[idx + 3]

                self._decode_stride(
                    scores, bboxes, None, stride, scale,
                    orig_w, orig_h, all_boxes, all_scores, all_kps,
                )
        else:
            logger.warning("Unexpected SCRFD output count: %d", num_outputs)
            return []

        if not all_boxes:
            return []

        boxes_arr = np.array(all_boxes)
        scores_arr = np.array(all_scores)

        # NMS
        nms_boxes = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in all_boxes]
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores_arr.tolist(),
            CFG.scrfd_conf_threshold,
            CFG.scrfd_iou_threshold,
        )

        results: list[FaceDetection] = []
        if len(indices) > 0:
            for idx in indices:
                i = int(idx)
                x1, y1, x2, y2 = boxes_arr[i].astype(int)
                face_w, face_h = x2 - x1, y2 - y1

                # Skip tiny faces
                if face_w < CFG.min_face_size_px or face_h < CFG.min_face_size_px:
                    continue

                if all_kps and i < len(all_kps):
                    lms = np.array(all_kps[i]).reshape(5, 2)
                    aligned = self._aligner.align(source_image, lms)
                else:
                    # Fallback: simple crop + resize if no landmarks
                    face_crop = source_image[
                        max(0, y1): min(orig_h, y2),
                        max(0, x1): min(orig_w, x2),
                    ]
                    aligned = cv2.resize(face_crop, (112, 112))
                    lms = np.zeros((5, 2), dtype=np.float32)

                results.append(
                    FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(scores_arr[i]),
                        landmarks=lms,
                        aligned_face=aligned,
                    )
                )
        return results

    def _decode_stride(
        self,
        scores: np.ndarray,
        bboxes: np.ndarray,
        kps: np.ndarray | None,
        stride: int,
        scale: float,
        orig_w: int,
        orig_h: int,
        all_boxes: list,
        all_scores: list,
        all_kps: list,
    ) -> None:
        """Decode detections from a single feature map stride."""
        scores = scores.reshape(-1)
        mask = scores >= CFG.scrfd_conf_threshold
        if not np.any(mask):
            return

        feat_h = CFG.scrfd_input_size // stride
        feat_w = CFG.scrfd_input_size // stride

        anchor_centers = np.stack(
            np.mgrid[:feat_h, :feat_w][::-1], axis=-1
        ).reshape(-1, 2) * stride

        # Handle 2-anchor case
        num_anchors = scores.shape[0] // (feat_h * feat_w)
        if num_anchors > 1:
            anchor_centers = np.tile(anchor_centers, (num_anchors, 1))

        bboxes = bboxes.reshape(-1, 4)
        filtered_indices = np.where(mask)[0]

        for i in filtered_indices:
            cx, cy = anchor_centers[i]
            dx1, dy1, dx2, dy2 = bboxes[i] * stride

            x1 = max(0, (cx - dx1) / scale)
            y1 = max(0, (cy - dy1) / scale)
            x2 = min(orig_w, (cx + dx2) / scale)
            y2 = min(orig_h, (cy + dy2) / scale)

            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(float(scores[i]))

            if kps is not None:
                kps_reshaped = kps.reshape(-1, 10)
                pts = kps_reshaped[i]
                landmarks = []
                for j in range(5):
                    lx = (cx + pts[j * 2] * stride) / scale
                    ly = (cy + pts[j * 2 + 1] * stride) / scale
                    landmarks.extend([lx, ly])
                all_kps.append(landmarks)
