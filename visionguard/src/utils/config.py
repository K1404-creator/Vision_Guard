"""
config.py
---------
Centralised configuration for the VisionGuard system.

Keeping all tunable parameters in one place satisfies the
'maintainability' and 'configurability' non-functional requirements:
changing a threshold or a path never requires touching business logic.
"""

import os

# ---------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
LOG_DIR = os.path.join(DATA_DIR, "logs")
SAMPLE_MEDIA_DIR = os.path.join(DATA_DIR, "sample_media")
DB_PATH = os.path.join(DATA_DIR, "visionguard.db")

# ---------------------------------------------------------------------
# Face recognition module
# ---------------------------------------------------------------------
FACE_MATCH_TOLERANCE = 0.5          # lower = stricter match
FACE_DETECTION_MODEL = "hog"        # 'hog' (CPU) or 'cnn' (GPU)
FACE_RESIZE_SCALE = 0.25            # downscale factor for faster detection

# ---------------------------------------------------------------------
# Object / motion detection module
# ---------------------------------------------------------------------
MOTION_MIN_AREA = 900               # minimum contour area (pixels) to count as motion
MOTION_BLUR_KERNEL = (21, 21)
MOTION_THRESHOLD = 25
OBJECT_CONFIDENCE_THRESHOLD = 0.5
OBJECT_NMS_THRESHOLD = 0.4

# ---------------------------------------------------------------------
# Analytics / reporting module
# ---------------------------------------------------------------------
REPORT_OUTPUT_DIR = os.path.join(BASE_DIR, "reports")
ALERT_COOLDOWN_SECONDS = 10         # avoid duplicate alerts within this window

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOG_DIR, "visionguard.log")

# ---------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------
# In production these would come from environment variables / a secrets
# manager rather than being hard-coded.
ADMIN_PIN_HASH_ENV_VAR = "VISIONGUARD_ADMIN_PIN_HASH"

for _dir in (DATA_DIR, KNOWN_FACES_DIR, LOG_DIR, SAMPLE_MEDIA_DIR, REPORT_OUTPUT_DIR):
    os.makedirs(_dir, exist_ok=True)
