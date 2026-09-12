"""
enroll.py
---------
Handles enrollment of new persons into the face database:
  1. Load an image containing exactly one clear face.
  2. Compute its 128-d face encoding (via face_recognition / dlib).
  3. Persist the person + encoding through DBManager.

This is the 'write' half of the Face Detection & Recognition module;
recognize.py is the 'read' half used at inference time.
"""

import os

import cv2
import face_recognition

from src.database.db_manager import DBManager
from src.utils.config import KNOWN_FACES_DIR
from src.utils.helpers import VisionGuardError, get_logger, safe_call

logger = get_logger(__name__)


class NoFaceFoundError(VisionGuardError):
    """Raised when zero faces are detected in an enrollment image."""


class MultipleFacesFoundError(VisionGuardError):
    """Raised when more than one face is detected in an enrollment image."""


class FaceEnroller:
    def __init__(self, db: DBManager = None):
        self.db = db or DBManager()

    @safe_call(default=False)
    def enroll_from_image(self, name: str, image_path: str) -> bool:
        """Enrolls `name` using the face found in `image_path`. Returns True on success."""
        if not os.path.exists(image_path):
            raise VisionGuardError(f"Image not found: {image_path}")

        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            raise NoFaceFoundError(f"No face detected in {image_path}")
        if len(face_locations) > 1:
            raise MultipleFacesFoundError(
                f"Expected exactly one face in {image_path}, found {len(face_locations)}"
            )

        encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]

        person_id = self.db.add_person(name)
        self.db.add_face_encoding(person_id, encoding)
        logger.info("Enrolled '%s' from %s", name, image_path)
        return True

    def enroll_from_webcam(self, name: str, camera_index: int = 0) -> bool:
        """Captures a single frame from the webcam and enrolls it."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise VisionGuardError("Could not access camera for enrollment")

        logger.info("Press SPACE to capture enrollment photo, ESC to cancel.")
        captured_path = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                cv2.imshow("Enroll - press SPACE to capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                if key == 32:  # SPACE
                    captured_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                    cv2.imwrite(captured_path, frame)
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if captured_path is None:
            return False
        return self.enroll_from_image(name, captured_path)


def bulk_enroll_from_directory(directory: str = KNOWN_FACES_DIR, db: DBManager = None) -> dict:
    """
    Enrolls every '<name>.jpg/png' file found in `directory`.
    Returns a summary dict {filename: success_bool}.
    """
    enroller = FaceEnroller(db)
    results = {}
    for filename in os.listdir(directory):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        name = os.path.splitext(filename)[0]
        results[filename] = enroller.enroll_from_image(name, os.path.join(directory, filename))
    return results


if __name__ == "__main__":
    summary = bulk_enroll_from_directory()
    for fname, ok in summary.items():
        print(f"{fname}: {'OK' if ok else 'FAILED'}")
