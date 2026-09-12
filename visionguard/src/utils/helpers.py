"""
helpers.py
----------
Small, reusable utility functions shared across modules, plus the
custom exception hierarchy used for the system's error-handling
strategy (see report section 'Non-Functional Requirements').
"""

import logging
import os
import time
from functools import wraps

from src.utils.config import LOG_FILE, LOG_LEVEL


# ---------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------
class VisionGuardError(Exception):
    """Base class for all application-specific errors."""


class CameraNotAvailableError(VisionGuardError):
    """Raised when a video source cannot be opened."""


class NoFaceEnrolledError(VisionGuardError):
    """Raised when recognition is attempted with an empty face database."""


class InvalidFrameError(VisionGuardError):
    """Raised when a frame captured from the source is empty/corrupt."""


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Returns a module-scoped logger writing to both console and a
    rotating log file, fulfilling the 'logging & monitoring'
    non-functional requirement.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ---------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------
def timed(func):
    """Logs the execution time of a function -- used to monitor performance
    (the 'Performance' non-functional requirement) of the hot paths
    (frame processing, recognition, detection)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("%s executed in %.2f ms", func.__name__, elapsed_ms)
        return result

    return wrapper


def safe_call(default=None):
    """
    Decorator implementing the system's defensive error-handling
    strategy: unexpected exceptions are logged and a safe default is
    returned instead of crashing the whole pipeline.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except VisionGuardError as exc:
                logger.warning("Handled application error in %s: %s", func.__name__, exc)
                return default
            except Exception as exc:  # noqa: BLE001 - last line of defence
                logger.error("Unexpected error in %s: %s", func.__name__, exc, exc_info=True)
                return default

        return wrapper

    return decorator


def ensure_directory(path: str) -> str:
    """Creates `path` if missing and returns it (idempotent helper)."""
    os.makedirs(path, exist_ok=True)
    return path
