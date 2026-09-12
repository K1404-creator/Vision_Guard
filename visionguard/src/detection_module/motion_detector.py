"""
motion_detector.py
-------------------
Lightweight background-subtraction based motion detector. This runs
far cheaper than the DNN object detector, so it is used as a
first-stage filter: the expensive object/face pipelines only need to
run on frames where motion was actually observed, which is the
system's main performance optimisation (see NFR: Performance).
"""

import cv2

from src.utils.config import MOTION_BLUR_KERNEL, MOTION_MIN_AREA, MOTION_THRESHOLD
from src.utils.helpers import get_logger, safe_call, timed

logger = get_logger(__name__)


class MotionDetector:
    def __init__(self):
        self.background = None

    def reset(self):
        self.background = None

    @timed
    @safe_call(default=(False, []))
    def detect(self, frame):
        """
        Compares `frame` against a running background model.

        Returns (motion_detected: bool, regions: list of (x, y, w, h)).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, MOTION_BLUR_KERNEL, 0)

        if self.background is None:
            self.background = gray.astype("float")
            return False, []

        cv2.accumulateWeighted(gray, self.background, 0.5)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        thresh = cv2.threshold(delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # cv2.findContours returns (contours, hierarchy) on OpenCV 4.x and
        # (image, contours, hierarchy) on OpenCV 3.x -- handle both without
        # requiring the extra 'imutils' dependency.
        found = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]

        regions = []
        for c in contours:
            if cv2.contourArea(c) < MOTION_MIN_AREA:
                continue
            regions.append(cv2.boundingRect(c))  # (x, y, w, h)

        return (len(regions) > 0), regions
