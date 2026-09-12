"""
main.py
-------
Entry point for VisionGuard. Wires together the three functional
modules into a single real-time pipeline:

    Camera / Video File
            |
            v
    [Motion Detector]  --- no motion --> (skip heavy processing, loop)
            |
       motion found
            |
            v
    [Face Recognizer]  +  [Object Detector]
            |
            v
    [Event Logger] -> SQLite -> [Report Generator] (on demand)

Run with:
    python -m src.main --source 0                 # webcam
    python -m src.main --source data/sample.mp4    # video file
    python -m src.main --report                    # just generate a report and exit
"""

import argparse
import time

import cv2

from src.analytics_module.logger import EventLogger
from src.analytics_module.report_generator import ReportGenerator
from src.database.db_manager import DBManager
from src.detection_module.motion_detector import MotionDetector
from src.detection_module.object_detector import ModelWeightsMissingError, ObjectDetector
from src.face_module.recognize import FaceRecognizer
from src.utils.helpers import CameraNotAvailableError, get_logger

logger = get_logger(__name__)


def draw_face_box(frame, name, confidence, box):
    top, right, bottom, left = box
    color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_object_box(frame, label, confidence, box):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)


def run_pipeline(source, display: bool = True, enable_objects: bool = True):
    db = DBManager()
    motion_detector = MotionDetector()
    face_recognizer = FaceRecognizer(db)
    event_logger = EventLogger(db)

    object_detector = None
    if enable_objects:
        try:
            object_detector = ObjectDetector()
        except ModelWeightsMissingError as exc:
            logger.warning("Object detector disabled: %s", exc)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise CameraNotAvailableError(f"Could not open video source: {source}")

    logger.info("VisionGuard pipeline started on source=%s", source)
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("End of stream reached.")
                break
            frame_count += 1

            motion_found, regions = motion_detector.detect(frame)
            if motion_found:
                event_logger.log_motion_event(len(regions), frame=frame, source=str(source))

                for face in face_recognizer.recognize_faces(frame):
                    draw_face_box(frame, face["name"], face["confidence"], face["box"])
                    event_logger.log_face_event(face["name"], face["confidence"], frame=frame,
                                                 source=str(source))

                if object_detector is not None:
                    for obj in object_detector.detect(frame):
                        draw_object_box(frame, obj["label"], obj["confidence"], obj["box"])
                        event_logger.log_object_event(obj["label"], obj["confidence"], frame=frame,
                                                       source=str(source))

            if display:
                cv2.imshow("VisionGuard", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        logger.info("Pipeline stopped after %d frames.", frame_count)


def main():
    parser = argparse.ArgumentParser(description="VisionGuard - CV Surveillance & Analytics")
    parser.add_argument("--source", default=0, help="Camera index or path to a video file")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no GUI window)")
    parser.add_argument("--no-objects", action="store_true", help="Disable object detection")
    parser.add_argument("--report", action="store_true", help="Generate a report and exit")
    args = parser.parse_args()

    if args.report:
        rg = ReportGenerator()
        result = rg.generate_all()
        print("Report summary:", result["summary"])
        print("Charts written to:", result["event_distribution_chart"], result["top_persons_chart"])
        return

    source = int(args.source) if str(args.source).isdigit() else args.source
    run_pipeline(source, display=not args.no_display, enable_objects=not args.no_objects)


if __name__ == "__main__":
    main()
