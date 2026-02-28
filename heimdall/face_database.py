"""
Face Database — In-memory NumPy + FAISS persistence.

Strategy for a small family (≤ 50 people, ≤ 500 embeddings):
  • Primary lookup: NumPy dot product (< 0.01 ms for 500 vectors)
  • Persistent storage: FAISS IndexFlatIP saved to disk
  • Metadata: JSON file mapping index positions → person names

Why not a full vector DB?
  For < 1000 vectors, NumPy is faster than any DB overhead.
  FAISS is used purely as a serialization format + future scalability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CFG

logger = logging.getLogger("heimdall.face_database")


@dataclass
class MatchResult:
    """Result of a face similarity query."""

    name: str
    similarity: float
    category: str  # "FAMILY", "UNCERTAIN", "STRANGER"


@dataclass
class _PersonEntry:
    """Internal: stores all embeddings for one person."""

    name: str
    embeddings: list[np.ndarray] = field(default_factory=list)


class FaceDatabase:
    """
    In-memory face embedding database with FAISS persistence.

    All biometric data stays 100% local.
    """

    def __init__(
        self,
        faiss_path: str | None = None,
        meta_path: str | None = None,
    ) -> None:
        self._faiss_path = faiss_path or CFG.faiss_index_path
        self._meta_path = meta_path or CFG.family_meta_path
        self._persons: dict[str, _PersonEntry] = {}

        # Flattened gallery for fast numpy search
        self._gallery_matrix: np.ndarray | None = None  # (N, 512)
        self._gallery_labels: list[str] = []             # length N

        self._dirty = False  # True if in-memory state differs from disk

    # ── Public API ────────────────────────────────────────────────────
    def load(self) -> None:
        """Load embeddings from disk (FAISS index + JSON metadata)."""
        meta_file = Path(self._meta_path)
        faiss_file = Path(self._faiss_path)

        if not meta_file.exists() or not faiss_file.exists():
            logger.info("No existing database found — starting fresh.")
            return

        try:
            import faiss  # type: ignore[import-untyped]

            index = faiss.read_index(str(faiss_file))
            n = index.ntotal
            if n == 0:
                logger.info("FAISS index is empty.")
                return

            # Reconstruct all vectors
            vectors = np.zeros((n, CFG.embedding_dim), dtype=np.float32)
            for i in range(n):
                vectors[i] = index.reconstruct(i)

            # Load metadata
            with open(meta_file, "r") as f:
                meta = json.load(f)

            # Rebuild in-memory structures
            self._persons.clear()
            labels = meta.get("labels", [])
            for i, name in enumerate(labels):
                if name not in self._persons:
                    self._persons[name] = _PersonEntry(name=name)
                self._persons[name].embeddings.append(vectors[i])

            self._rebuild_gallery()
            logger.info(
                "Loaded %d embeddings for %d persons from disk.",
                n,
                len(self._persons),
            )

        except ImportError:
            logger.warning("faiss not installed — loading from numpy fallback.")
            self._load_numpy_fallback()

    def save(self) -> None:
        """Persist current embeddings to disk."""
        if self._gallery_matrix is None or len(self._gallery_labels) == 0:
            logger.info("Nothing to save — database is empty.")
            return

        try:
            import faiss  # type: ignore[import-untyped]

            dim = CFG.embedding_dim
            index = faiss.IndexFlatIP(dim)  # Inner Product (= cosine for L2-normed)
            index.add(self._gallery_matrix.astype(np.float32))

            faiss.write_index(index, str(self._faiss_path))
        except ImportError:
            # Fallback: save as .npy
            np.save(self._faiss_path.replace(".faiss", ".npy"), self._gallery_matrix)

        meta = {"labels": self._gallery_labels}
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._dirty = False
        logger.info(
            "Saved %d embeddings to %s", len(self._gallery_labels), self._faiss_path
        )

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """
        Add an embedding for a person.

        Automatically limits to ``max_gallery_per_person`` embeddings,
        dropping the oldest if exceeded.
        """
        if name not in self._persons:
            self._persons[name] = _PersonEntry(name=name)

        entry = self._persons[name]
        entry.embeddings.append(embedding.astype(np.float32))

        # Cap the number of embeddings per person
        if len(entry.embeddings) > CFG.max_gallery_per_person:
            entry.embeddings = entry.embeddings[-CFG.max_gallery_per_person :]

        self._rebuild_gallery()
        self._dirty = True
        logger.info(
            "Enrolled embedding for '%s' (total: %d)", name, len(entry.embeddings)
        )

    def query(self, probe: np.ndarray) -> MatchResult:
        """
        Find the closest match for a probe embedding.

        Parameters
        ----------
        probe : np.ndarray
            512-D L2-normalized embedding.

        Returns
        -------
        MatchResult
            Best match with similarity score and category.
        """
        if self._gallery_matrix is None or len(self._gallery_labels) == 0:
            return MatchResult(name="unknown", similarity=0.0, category="STRANGER")

        # Cosine similarity = dot product (both vectors are L2-normalized)
        similarities = self._gallery_matrix @ probe  # shape (N,)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_name = self._gallery_labels[best_idx]

        # Classify
        if best_sim >= CFG.tau_accept:
            category = "FAMILY"
        elif best_sim < CFG.tau_reject:
            category = "STRANGER"
            best_name = "unknown"
        else:
            category = "UNCERTAIN"

        return MatchResult(name=best_name, similarity=best_sim, category=category)

    def get_all_names(self) -> list[str]:
        """Return list of enrolled person names."""
        return list(self._persons.keys())

    def get_person_count(self) -> int:
        return len(self._persons)

    def get_total_embeddings(self) -> int:
        return sum(len(p.embeddings) for p in self._persons.values())

    # ── Private ───────────────────────────────────────────────────────
    def _rebuild_gallery(self) -> None:
        """Flatten all person embeddings into a single matrix for fast search."""
        all_embs: list[np.ndarray] = []
        all_labels: list[str] = []

        for name, entry in self._persons.items():
            for emb in entry.embeddings:
                all_embs.append(emb)
                all_labels.append(name)

        if all_embs:
            self._gallery_matrix = np.stack(all_embs, axis=0).astype(np.float32)
            self._gallery_labels = all_labels
        else:
            self._gallery_matrix = None
            self._gallery_labels = []

    def _load_numpy_fallback(self) -> None:
        """Load from .npy + .json when faiss is unavailable."""
        npy_path = self._faiss_path.replace(".faiss", ".npy")
        meta_file = Path(self._meta_path)

        if not Path(npy_path).exists() or not meta_file.exists():
            return

        vectors = np.load(npy_path)
        with open(meta_file, "r") as f:
            meta = json.load(f)

        labels = meta.get("labels", [])
        self._persons.clear()
        for i, name in enumerate(labels):
            if name not in self._persons:
                self._persons[name] = _PersonEntry(name=name)
            self._persons[name].embeddings.append(vectors[i])

        self._rebuild_gallery()
