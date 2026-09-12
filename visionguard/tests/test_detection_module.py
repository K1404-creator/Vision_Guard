"""
test_detection_module.py
-------------------------
Unit tests for MotionDetector and ObjectDetector.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.detection_module.motion_detector import MotionDetector


class TestMotionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = MotionDetector()

    def test_first_frame_initializes_background_without_motion(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        motion_found, regions = self.detector.detect(frame)
        self.assertFalse(motion_found)
        self.assertEqual(regions, [])

    def test_identical_frames_report_no_motion(self):
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        self.detector.detect(frame)  # seed background
        motion_found, regions = self.detector.detect(frame)
        self.assertFalse(motion_found)
        self.assertEqual(regions, [])

    def test_large_change_reports_motion(self):
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        self.detector.detect(blank)  # seed background

        changed = blank.copy()
        changed[50:150, 50:150] = 255  # large bright square = big frame delta

        motion_found, regions = self.detector.detect(changed)
        self.assertTrue(motion_found)
        self.assertGreaterEqual(len(regions), 1)

    def test_reset_clears_background_model(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.detector.detect(frame)
        self.assertIsNotNone(self.detector.background)
        self.detector.reset()
        self.assertIsNone(self.detector.background)


class TestObjectDetectorMissingWeights(unittest.TestCase):
    @patch("src.detection_module.object_detector.os.path.exists", return_value=False)
    def test_raises_when_weights_missing(self, _mock_exists):
        from src.detection_module.object_detector import ModelWeightsMissingError, ObjectDetector
        with self.assertRaises(ModelWeightsMissingError):
            ObjectDetector()


if __name__ == "__main__":
    unittest.main()
