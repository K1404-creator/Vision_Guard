"""
test_analytics_module.py
-------------------------
Unit tests for EventLogger (cooldown logic) and ReportGenerator
(summary correctness).
"""

import time
import unittest

import numpy as np

from src.analytics_module.logger import EventLogger
from src.analytics_module.report_generator import ReportGenerator
from src.database.db_manager import DBManager


class TestEventLogger(unittest.TestCase):
    def setUp(self):
        self.db = DBManager(db_path=":memory:")
        self.event_logger = EventLogger(self.db)
        self.frame = np.zeros((10, 10, 3), dtype=np.uint8)

    def test_logs_first_event(self):
        self.event_logger.log_face_event("alice", 0.9, frame=self.frame)
        events = self.db.fetch_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["label"], "alice")

    def test_cooldown_suppresses_duplicate_events(self):
        self.event_logger.log_face_event("alice", 0.9, frame=self.frame)
        self.event_logger.log_face_event("alice", 0.9, frame=self.frame)  # within cooldown window
        events = self.db.fetch_events()
        self.assertEqual(len(events), 1)

    def test_different_labels_not_suppressed(self):
        self.event_logger.log_face_event("alice", 0.9, frame=self.frame)
        self.event_logger.log_face_event("bob", 0.8, frame=self.frame)
        events = self.db.fetch_events()
        self.assertEqual(len(events), 2)


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.db = DBManager(db_path=":memory:")
        self.db.log_event("face_recognized", label="alice", confidence=0.9)
        self.db.log_event("face_recognized", label="alice", confidence=0.95)
        self.db.log_event("motion", label="1 region(s)")
        self.report_generator = ReportGenerator(self.db)

    def test_summarize_counts_by_type(self):
        summary = self.report_generator.summarize()
        self.assertEqual(summary["total_events"], 3)
        self.assertEqual(summary["by_type"]["face_recognized"], 2)
        self.assertEqual(summary["by_type"]["motion"], 1)

    def test_summarize_top_persons(self):
        summary = self.report_generator.summarize()
        self.assertEqual(summary["top_persons"][0], ("alice", 2))


if __name__ == "__main__":
    unittest.main()
