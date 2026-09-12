# Problem Statement

Manual monitoring of physical spaces — offices, labs, entry points,
classrooms — for who enters, what objects are present, and when activity
occurs is time-consuming, error-prone, and does not scale. Simple CCTV
recording captures footage but provides no automatic understanding of
*who* or *what* was seen, and no easy way to review activity patterns
over time without manually scrubbing through hours of video.

VisionGuard addresses this by combining face recognition, object/motion
detection, and automatic event logging + reporting into a single
lightweight pipeline that runs on a standard laptop webcam, turning raw
video into structured, queryable, and visualized events.

---

## Scope of the Project

**In scope:**
- Real-time face detection and recognition against an enrolled database
  of known individuals.
- Real-time detection of common objects (person, car, bottle, chair,
  etc.) and motion events from a live camera or a recorded video file.
- Persistent event logging (SQLite) with cooldown-based de-duplication
  and snapshot capture.
- Automatic generation of summary statistics and charts from logged
  events.
- A command-line interface for enrollment, live monitoring, and
  report generation.

**Out of scope (documented as future enhancements):**
- A web-based dashboard / multi-user login system.
- Multi-camera synchronization.
- Cloud storage or remote streaming.
- Real-time alerting via SMS/email/push notification.

---

## Target Users

- **Students / educators** demonstrating applied computer vision concepts
  for a course project or lab.
- **Small offices, labs, or home users** who want a self-hosted,
  privacy-respecting alternative to commercial "smart camera" products
  (no footage leaves the local machine).
- **Developers** looking for a modular reference implementation that
  combines face recognition, object detection, and event analytics in
  one codebase.

---

## High-Level Features

1. **Face Detection & Recognition Module**
   - Enroll a person from a single photo or a webcam capture.
   - Recognize enrolled faces in real time with a confidence score.
   - Gracefully report "Unknown" for unrecognized faces instead of
     failing.

2. **Object & Motion Detection Module**
   - Lightweight motion detection acts as a first-stage filter so the
     more expensive recognition/detection models only run when there is
     actual activity in frame (performance optimisation).
   - MobileNet-SSD based object detection identifies 20 common object
     classes with bounding boxes and confidence scores.

3. **Analytics & Reporting Module**
   - Every recognized face, detected object, and motion burst is logged
     to a local SQLite database with a timestamp and snapshot image.
   - A reporting utility summarizes total events, breakdown by event
     type, and the most frequently seen people, and renders this as bar
     charts for easy interpretation.
