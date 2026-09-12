"""
test_face_module.py
--------------------
Unit tests for enrollment and recognition. Uses an in-memory SQLite
database (":memory:") so tests never touch real project data and can
run in isolation / CI.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.database.db_manager import DBManager
from src.face_module.enroll import FaceEnroller, MultipleFacesFoundError, NoFaceFoundError
from src.face_module.recognize import FaceRecognizer
from src.utils.helpers import NoFaceEnrolledError


class TestFaceEnrollment(unittest.TestCase):
    def setUp(self):
        self.db = DBManager(db_path=":memory:")
        self.enroller = FaceEnroller(self.db)

    @patch("src.face_module.enroll.face_recognition")
    @patch("src.face_module.enroll.os.path.exists", return_value=True)
    def test_enroll_success(self, _mock_exists, mock_fr):
        mock_fr.load_image_file.return_value = np.zeros((100, 100, 3))
        mock_fr.face_locations.return_value = [(0, 10, 10, 0)]
        mock_fr.face_encodings.return_value = [np.random.rand(128)]

        result = self.enroller.enroll_from_image("alice", "fake_path.jpg")
        self.assertTrue(result)

        encodings = self.db.get_all_encodings()
        self.assertEqual(len(encodings), 1)
        self.assertEqual(encodings[0][0], "alice")

    @patch("src.face_module.enroll.face_recognition")
    @patch("src.face_module.enroll.os.path.exists", return_value=True)
    def test_enroll_no_face_returns_false(self, _mock_exists, mock_fr):
        mock_fr.load_image_file.return_value = np.zeros((100, 100, 3))
        mock_fr.face_locations.return_value = []

        result = self.enroller.enroll_from_image("bob", "fake_path.jpg")
        # safe_call swallows the NoFaceFoundError and returns default=False
        self.assertFalse(result)

    @patch("src.face_module.enroll.face_recognition")
    @patch("src.face_module.enroll.os.path.exists", return_value=True)
    def test_enroll_multiple_faces_returns_false(self, _mock_exists, mock_fr):
        mock_fr.load_image_file.return_value = np.zeros((100, 100, 3))
        mock_fr.face_locations.return_value = [(0, 10, 10, 0), (20, 30, 30, 20)]

        result = self.enroller.enroll_from_image("crowd", "fake_path.jpg")
        self.assertFalse(result)


class TestFaceRecognition(unittest.TestCase):
    def setUp(self):
        self.db = DBManager(db_path=":memory:")

    def test_recognize_raises_when_no_faces_enrolled(self):
        recognizer = FaceRecognizer(self.db)
        # safe_call decorator converts the raised NoFaceEnrolledError into default=[]
        result = recognizer.recognize_faces(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(result, [])

    @patch("src.face_module.recognize.face_recognition")
    def test_recognize_matches_known_face(self, mock_fr):
        # Enroll one known encoding directly through the DB layer
        person_id = self.db.add_person("alice")
        known_vector = np.random.rand(128)
        self.db.add_face_encoding(person_id, known_vector)

        recognizer = FaceRecognizer(self.db)

        mock_fr.face_locations.return_value = [(0, 10, 10, 0)]
        mock_fr.face_encodings.return_value = [known_vector]  # identical -> distance 0
        mock_fr.face_distance.return_value = np.array([0.0])

        result = recognizer.recognize_faces(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "alice")


if __name__ == "__main__":
    unittest.main()
