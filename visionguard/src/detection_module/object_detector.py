"""
object_detector.py
-------------------
Wraps an OpenCV DNN (MobileNet-SSD, Caffe weights) to perform generic
object detection frame-by-frame. Kept independent of the face module
so either can be enabled/disabled without affecting the other
(modularity / single-responsibility).
"""

import os

import cv2
import numpy as np

from src.utils.config import OBJECT_CONFIDENCE_THRESHOLD, OBJECT_NMS_THRESHOLD
from src.utils.helpers import VisionGuardError, get_logger, safe_call, timed

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class ModelWeightsMissingError(VisionGuardError):
    """Raised when the pretrained model files are not present on disk."""


class ObjectDetector:
    def __init__(self, prototxt: str = PROTOTXT_PATH, weights: str = WEIGHTS_PATH):
        if not (os.path.exists(prototxt) and os.path.exists(weights)):
            raise ModelWeightsMissingError(
                "MobileNet-SSD weights not found. Run "
                "`scripts/download_models.sh` to fetch them (see README)."
            )
        self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        logger.info("Object detector model loaded from %s", weights)

    @timed
    @safe_call(default=[])
    def detect(self, frame) -> list:
        """
        Runs object detection on a single BGR frame.

        Returns a list of dicts:
            {"label": str, "confidence": float, "box": (x1, y1, x2, y2)}
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < OBJECT_CONFIDENCE_THRESHOLD:
                continue
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            results.append({"label": label, "confidence": round(confidence, 3),
                             "box": (int(x1), int(y1), int(x2), int(y2))})
        return results
