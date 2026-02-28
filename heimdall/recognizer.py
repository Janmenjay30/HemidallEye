"""
Face Recognizer — ArcFace (w600k_r50) via ONNX Runtime.

Responsibilities:
  • Accept an aligned 112×112 face image
  • Produce a 512-D L2-normalized embedding
  • Embeddings are suitable for Cosine Similarity comparison
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import onnxruntime as ort

from .config import CFG, ARCFACE_MODEL_PATH
from .vram_manager import create_onnx_session, session_uses_cuda


def _ort_type_to_numpy(ort_type: str) -> np.dtype:
    """Convert ONNX Runtime type string to numpy dtype."""
    mapping = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}
    return mapping.get(ort_type, np.float32)

logger = logging.getLogger("heimdall.recognizer")


class FaceRecognizer:
    """
    ArcFace-R50 (WebFace600K pretrained) embedding extractor.

    VRAM budget: ~500 MB (FP16).
    Output: 512-D L2-normalized vector.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or str(ARCFACE_MODEL_PATH)
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._input_dtype: np.dtype = np.float32

    # ── Lifecycle ─────────────────────────────────────────────────────
    def load(self) -> None:
        """Load ArcFace model onto GPU."""
        logger.info("Loading ArcFace from %s", self._model_path)
        self._session = create_onnx_session(
            self._model_path,
            gpu_mem_limit_mb=600,
        )
        meta = self._session.get_inputs()[0]
        self._input_name = meta.name
        self._input_dtype = _ort_type_to_numpy(meta.type)
        on_gpu = session_uses_cuda(self._session)
        logger.info("ArcFace ready \u2014 input: %s  dtype=%s  GPU=%s", self._input_name, self._input_dtype, on_gpu)

    def unload(self) -> None:
        del self._session
        self._session = None
        logger.info("ArcFace unloaded")

    # ── Inference ─────────────────────────────────────────────────────
    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract a 512-D embedding from an aligned 112×112 BGR face.

        Parameters
        ----------
        aligned_face : np.ndarray
            112×112×3 BGR aligned face image.

        Returns
        -------
        np.ndarray
            512-D L2-normalized embedding (float32).
        """
        if self._session is None:
            raise RuntimeError("ArcFace not loaded. Call .load() first.")

        blob = self._preprocess(aligned_face)
        outputs = self._session.run(None, {self._input_name: blob})
        embedding = outputs[0][0]  # shape (512,)

        # L2-normalize (should already be close to unit norm, but ensure it)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def get_embeddings_batch(
        self, aligned_faces: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Extract embeddings for multiple faces.

        For a small number of faces (typical CCTV scenario), sequential
        processing is fine. Batching is future-proofing.
        """
        return [self.get_embedding(face) for face in aligned_faces]

    # ── Preprocessing ─────────────────────────────────────────────────
    def _preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Normalize aligned face for ArcFace input.

        Standard ArcFace preprocessing:
          • BGR → RGB
          • Subtract 127.5, divide by 127.5 → range [-1, 1]
          • HWC → CHW, add batch dim
        """
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # add batch dim

        img = img.astype(self._input_dtype)

        return img
