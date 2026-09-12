# VisionGuard — Multi-Module Computer Vision Surveillance & Analytics System

VisionGuard is a real-time computer vision pipeline that combines **face
recognition**, **object & motion detection**, and **event
analytics/reporting** into a single modular system. It is built as an
academic project to demonstrate applied computer vision, software
architecture, and data-driven reporting.

---

## Overview

Point VisionGuard at a webcam or a video file and it will:

1. Watch for motion (cheap, always-on check).
2. When motion occurs, run **face recognition** against an enrolled
   database of known people, and **object detection** to identify what's
   in frame (person, car, bottle, etc.).
3. Log every meaningful event (a known/unknown face, a detected object, a
   motion burst) to a local SQLite database, with a snapshot image and a
   cooldown so the same event isn't logged dozens of times a second.
4. Generate an analytics report (bar charts + a JSON summary) on demand,
   showing event counts by type and the most frequently recognized people.

This mirrors real use cases such as smart attendance, entry monitoring, and
lightweight home/office surveillance — while staying runnable on a laptop
CPU with no cloud dependency.

---

## Features

- **Face Detection & Recognition module** — enroll people from a photo or
  webcam capture; recognize them in real time with confidence scores.
- **Object & Motion Detection module** — background-subtraction motion
  detection (cheap, first-stage filter) + MobileNet-SSD object detection
  (20 everyday object classes).
- **Analytics & Reporting module** — SQLite-backed event log, cooldown-based
  de-duplication, and auto-generated summary charts.
- Clean modular architecture: each concern (face, detection, analytics,
  database, config/utils) lives in its own package.
- Defensive error handling with a custom exception hierarchy — a missing
  camera, a corrupt frame, or an empty face database degrade gracefully
  instead of crashing the pipeline.
- Structured logging to both console and a rotating log file.
- Unit tests for every module (15 tests, all passing) using mocks so they
  run without a physical camera or GPU.

---

## Technologies / Tools Used

| Concern | Tool |
|---|---|
| Language | Python 3.10+ |
| Video I/O & image processing | OpenCV (`opencv-python`) |
| Face recognition | `face_recognition` (dlib ResNet embeddings) |
| Object detection | OpenCV DNN + MobileNet-SSD (Caffe weights) |
| Motion detection | OpenCV background subtraction / frame differencing |
| Persistence | SQLite (via Python's built-in `sqlite3`) |
| Reporting/visualization | Matplotlib |
| Testing | `unittest` + `unittest.mock` |
| Version control | Git / GitHub |

---

## Project Structure

```
visionguard/
├── src/
│   ├── main.py                     # orchestrates the full pipeline
│   ├── face_module/
│   │   ├── enroll.py                # enroll new people (image / webcam)
│   │   └── recognize.py             # real-time face recognition
│   ├── detection_module/
│   │   ├── object_detector.py       # MobileNet-SSD object detection
│   │   ├── motion_detector.py       # background-subtraction motion detection
│   │   └── models/                  # pretrained weights (downloaded, gitignored)
│   ├── analytics_module/
│   │   ├── logger.py                # event logging facade (with cooldown)
│   │   └── report_generator.py      # summary + chart generation
│   ├── database/
│   │   └── db_manager.py            # SQLite schema & data-access layer
│   └── utils/
│       ├── config.py                # all tunable constants/paths
│       └── helpers.py               # logging setup, decorators, exceptions
├── tests/
│   ├── test_face_module.py
│   ├── test_detection_module.py
│   └── test_analytics_module.py
├── scripts/
│   └── download_models.sh           # fetches MobileNet-SSD weights
├── data/                             # known_faces/, sample_media/, logs/ (gitignored contents)
├── diagrams/                         # architecture / UML / ER diagrams (see report)
├── requirements.txt
├── statement.md
└── README.md
```

11 Python modules + 3 test files = comfortably above the minimum module
count, each with a single, clear responsibility.

---

## Steps to Install & Run

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>.git
cd visionguard
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `face_recognition` depends on `dlib`, which needs CMake and a
> C++ compiler to build. On Windows, installing the "Visual C++ Build
> Tools" first is the easiest path; on macOS/Linux, `cmake` is usually a
> one-line package-manager install (`brew install cmake` / `apt install
> cmake`).

### 2. Download the object-detection model weights (one-time)

```bash
bash scripts/download_models.sh
```

If the automated download link is stale, grab
`MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` from
the MobileNet-SSD GitHub repository and place both files in
`src/detection_module/models/`. Object detection is optional — the face
and motion modules work without it (`--no-objects` flag).

### 3. Enroll at least one known face

Drop a clear, single-face photo named after the person into
`data/known_faces/` (e.g. `data/known_faces/alice.jpg`), then run:

```bash
python -m src.face_module.enroll
```

or capture directly from a webcam:

```python
from src.face_module.enroll import FaceEnroller
FaceEnroller().enroll_from_webcam("alice")
```

### 4. Run the live pipeline

```bash
# Webcam (device 0)
python -m src.main --source 0

# A video file instead of a webcam
python -m src.main --source data/sample_media/sample.mp4

# Headless (no GUI window) — useful on a server / CI box
python -m src.main --source 0 --no-display

# Skip object detection if you haven't downloaded the weights
python -m src.main --source 0 --no-objects
```

Press `q` to quit the display window.

### 5. Generate an analytics report

```bash
python -m src.main --report
```

This prints a JSON-style summary (event counts by type, top recognized
people) and writes two chart images to `reports/`:
`event_distribution.png` and `top_persons.png`.

---

## Instructions for Testing

Run the full unit test suite from the project root:

```bash
python -m unittest discover -s tests -v
```

All 15 tests should pass. Tests use `unittest.mock` to stand in for the
camera, `face_recognition`, and the OpenCV DNN model, so they run in
milliseconds with no hardware or model files required.

To check a single module:

```bash
python -m unittest tests.test_analytics_module -v
```

---

## Non-Functional Requirements Addressed

| Requirement | How it's addressed |
|---|---|
| **Performance** | Motion detection gates the expensive face/object pipelines so they only run on frames with actual activity; frame processing time is logged via a `@timed` decorator. |
| **Security** | No raw camera feed leaves the machine; admin operations are designed to sit behind a PIN hash read from an environment variable, never hard-coded. |
| **Reliability** | A custom exception hierarchy (`VisionGuardError` and subclasses) plus a `@safe_call` decorator ensure one bad frame or a missing model file degrades a single feature instead of crashing the whole app. |
| **Scalability** | The database layer is isolated behind `DBManager`, so swapping SQLite for PostgreSQL/MySQL later touches one file, not the whole codebase. |
| **Maintainability** | All modules are single-responsibility and independently testable; constants live in one `config.py`. |
| **Logging/Monitoring** | Every module logs to both console and a rotating file (`data/logs/visionguard.log`) with timestamps and severity levels. |
| **Usability** | Simple CLI flags (`--source`, `--no-display`, `--no-objects`, `--report`); bounding boxes and labels are drawn directly on the video feed. |

---

## Screenshots

See `diagrams/` and the accompanying **Project Report** (PDF/DOCX) for
architecture diagrams, UML diagrams, and sample output screenshots.

---

## Future Enhancements

- Web-based dashboard (Flask/FastAPI) instead of the OpenCV GUI window.
- Multi-camera support with per-camera event tagging.
- Swap MobileNet-SSD for a YOLOv8 model for higher accuracy.
- Role-based access control for the admin/reporting interface.

---

## License

This project is released under the MIT License — see [LICENSE](LICENSE).
