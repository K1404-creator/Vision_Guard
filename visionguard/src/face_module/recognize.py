"""
recognize.py
------------
Real-time face recognition against the enrolled face database.
Designed to be called frame-by-frame from main.py so it can be
combined with the detection and analytics modules in a single
video pipeline.
"""

import numpy as np
import face_recognition

from src.database.db_manager import DBManager
from src.utils.config import FACE_MATCH_TOLERANCE, FACE_RESIZE_SCALE
from src.utils.helpers import NoFaceEnrolledError, get_logger, safe_call, timed

logger = get_logger(__name__)


class FaceRecognizer:
    def __init__(self, db: DBManager = None):
        self.db = db or DBManager()
        self.known_names = []
        self.known_encodings = []
        self.refresh_known_faces()

    def refresh_known_faces(self) -> None:
        """Reloads the in-memory encoding cache from the database.
        Call this after new enrollments so recognition picks them up
        without restarting the whole system."""
        pairs = self.db.get_all_encodings()
        self.known_names = [p[0] for p in pairs]
        self.known_encodings = [np.array(p[1]) for p in pairs]
        logger.info("Loaded %d known face encodings", len(self.known_encodings))

    @timed
    @safe_call(default=[])
    def recognize_faces(self, frame) -> list:
        """
        Detects and identifies faces in a single BGR frame.

        Returns a list of dicts:
            {"name": str, "confidence": float, "box": (top, right, bottom, left)}
        `name` is "Unknown" when no match clears FACE_MATCH_TOLERANCE.
        """
        if not self.known_encodings:
            raise NoFaceEnrolledError("No faces enrolled yet; skipping recognition.")

        small_frame = frame[:, :, ::-1]  # BGR -> RGB
        if FACE_RESIZE_SCALE != 1.0:
            import cv2
            small_frame = cv2.resize(small_frame, (0, 0), fx=FACE_RESIZE_SCALE, fy=FACE_RESIZE_SCALE)

        locations = face_recognition.face_locations(small_frame)
        encodings = face_recognition.face_encodings(small_frame, locations)

        results = []
        for (top, right, bottom, left), face_encoding in zip(locations, encodings):
            distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            best_idx = int(np.argmin(distances)) if len(distances) else None

            name = "Unknown"
            confidence = 0.0
            if best_idx is not None and distances[best_idx] <= FACE_MATCH_TOLERANCE:
                name = self.known_names[best_idx]
                confidence = round(1 - distances[best_idx], 3)

            scale = 1 / FACE_RESIZE_SCALE
            box = (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))
            results.append({"name": name, "confidence": confidence, "box": box})

        return results
