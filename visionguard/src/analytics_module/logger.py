"""
logger.py (analytics_module)
-----------------------------
Central event-logging facade. The face and detection modules report
what they see here; this module decides what is worth persisting
(applying alert cooldown so the same event isn't logged 30 times a
second) and writes it to the database via DBManager.
"""

import os
import time
from datetime import datetime, timezone

import cv2

from src.database.db_manager import DBManager
from src.utils.config import ALERT_COOLDOWN_SECONDS, LOG_DIR
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class EventLogger:
    def __init__(self, db: DBManager = None):
        self.db = db or DBManager()
        self._last_logged = {}  # key -> last timestamp, for cooldown

    def _cooldown_ok(self, key: str) -> bool:
        now = time.time()
        last = self._last_logged.get(key, 0)
        if now - last >= ALERT_COOLDOWN_SECONDS:
            self._last_logged[key] = now
            return True
        return False

    def log_face_event(self, name: str, confidence: float, frame=None, source: str = "camera-0"):
        key = f"face:{name}"
        if not self._cooldown_ok(key):
            return
        event_type = "face_recognized" if name != "Unknown" else "unknown_face"
        image_path = self._save_snapshot(frame, prefix=event_type) if frame is not None else None
        self.db.log_event(event_type, label=name, confidence=confidence,
                           source=source, image_path=image_path)
        logger.info("Logged %s: %s (%.2f)", event_type, name, confidence)

    def log_object_event(self, label: str, confidence: float, frame=None, source: str = "camera-0"):
        key = f"object:{label}"
        if not self._cooldown_ok(key):
            return
        image_path = self._save_snapshot(frame, prefix="object") if frame is not None else None
        self.db.log_event("object_detected", label=label, confidence=confidence,
                           source=source, image_path=image_path)
        logger.info("Logged object_detected: %s (%.2f)", label, confidence)

    def log_motion_event(self, region_count: int, frame=None, source: str = "camera-0"):
        key = "motion"
        if not self._cooldown_ok(key):
            return
        image_path = self._save_snapshot(frame, prefix="motion") if frame is not None else None
        self.db.log_event("motion", label=f"{region_count} region(s)", source=source,
                           image_path=image_path)
        logger.info("Logged motion event with %d region(s)", region_count)

    @staticmethod
    def _save_snapshot(frame, prefix: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        path = os.path.join(LOG_DIR, filename)
        cv2.imwrite(path, frame)
        return path
