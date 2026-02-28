"""
Family Face Enrollment CLI — Register known faces into the database.

Usage:
    # Enroll from a directory of images
    python -m heimdall.enroll --name "Alice" --images ./photos/alice/

    # Enroll from a single image
    python -m heimdall.enroll --name "Bob" --image ./photos/bob.jpg

    # Enroll from webcam (take 5 snapshots)
    python -m heimdall.enroll --name "Charlie" --webcam --count 5

    # List enrolled persons
    python -m heimdall.enroll --list
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import CFG, PHOTOS_DIR
from .face_database import FaceDatabase
from .face_detector import FaceDetector
from .recognizer import FaceRecognizer

logger = logging.getLogger("heimdall.enroll")


def enroll_from_images(
    name: str,
    image_paths: list[Path],
    face_detector: FaceDetector,
    recognizer: FaceRecognizer,
    database: FaceDatabase,
) -> int:
    """
    Enroll a person from a list of image files.

    Returns the number of successfully enrolled embeddings.
    """
    enrolled = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read: %s", img_path)
            continue

        # Detect faces in the full image
        faces = face_detector.detect_faces(img)
        if not faces:
            logger.warning("No face found in: %s", img_path)
            continue

        # Use the most confident face
        best_face = max(faces, key=lambda f: f.confidence)

        # Extract embedding
        embedding = recognizer.get_embedding(best_face.aligned_face)

        # Enroll
        database.enroll(name, embedding)
        enrolled += 1
        logger.info("Enrolled from %s (conf=%.3f)", img_path.name, best_face.confidence)

    return enrolled


def enroll_from_webcam(
    name: str,
    count: int,
    face_detector: FaceDetector,
    recognizer: FaceRecognizer,
    database: FaceDatabase,
) -> int:
    """
    Enroll a person by capturing frames from the default webcam.

    Shows a preview window. Press SPACE to capture, ESC to cancel.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return 0

    enrolled = 0
    print(f"\nWebcam enrollment for '{name}' — need {count} captures.")
    print("Press SPACE to capture, ESC to cancel.\n")

    while enrolled < count:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(
            display,
            f"Enrolled: {enrolled}/{count} — SPACE=capture, ESC=cancel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Heimdall Enrollment", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            faces = face_detector.detect_faces(frame)
            if not faces:
                print("  No face detected — try again.")
                continue

            best_face = max(faces, key=lambda f: f.confidence)
            embedding = recognizer.get_embedding(best_face.aligned_face)
            database.enroll(name, embedding)
            enrolled += 1
            print(f"  Captured {enrolled}/{count} (conf={best_face.confidence:.3f})")

    cap.release()
    cv2.destroyAllWindows()
    return enrolled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heimdall — Family Face Enrollment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", type=str, help="Person's name to enroll")
    parser.add_argument(
        "--images", type=str, help="Directory containing face images"
    )
    parser.add_argument("--image", type=str, help="Single face image path")
    parser.add_argument(
        "--webcam", action="store_true", help="Enroll from webcam"
    )
    parser.add_argument(
        "--count", type=int, default=5, help="Number of webcam captures"
    )
    parser.add_argument(
        "--list", action="store_true", help="List enrolled persons and exit"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── Initialize components ─────────────────────────────────────
    database = FaceDatabase()
    database.load()

    if args.list:
        names = database.get_all_names()
        if not names:
            print("No persons enrolled yet.")
        else:
            print(f"\nEnrolled persons ({len(names)}):")
            for n in names:
                print(f"  • {n}")
            print(f"\nTotal embeddings: {database.get_total_embeddings()}")
        return

    if not args.name:
        parser.error("--name is required for enrollment")

    # Load models
    face_detector = FaceDetector()
    recognizer = FaceRecognizer()

    print("Loading models...")
    face_detector.load()
    recognizer.load()

    # ── Enroll ────────────────────────────────────────────────────
    if args.webcam:
        n = enroll_from_webcam(
            args.name, args.count, face_detector, recognizer, database
        )
    elif args.images:
        img_dir = Path(args.images)
        paths = sorted(
            p
            for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
        if not paths:
            print(f"No images found in {img_dir}")
            return
        n = enroll_from_images(args.name, paths, face_detector, recognizer, database)
    elif args.image:
        n = enroll_from_images(
            args.name, [Path(args.image)], face_detector, recognizer, database
        )
    else:
        parser.error("Specify --images, --image, or --webcam")
        return

    if n > 0:
        database.save()
        print(f"\n✓ Enrolled {n} embedding(s) for '{args.name}'")
    else:
        print(f"\n✗ No embeddings enrolled for '{args.name}'")

    # Cleanup
    face_detector.unload()
    recognizer.unload()


if __name__ == "__main__":
    main()
