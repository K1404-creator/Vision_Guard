"""
db_manager.py
-------------
Thin data-access layer around SQLite. All modules go through this
class rather than writing SQL directly -- this keeps the schema in one
place and makes the system easier to migrate to another database
engine later (Postgres/MySQL) without touching business logic.

Schema
------
persons(id, name, enrolled_on)
face_encodings(id, person_id -> persons.id, encoding BLOB)
events(id, event_type, label, confidence, source, image_path, created_at)
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from src.utils.config import DB_PATH
from src.utils.helpers import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    enrolled_on TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_encodings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL,
    encoding   TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,      -- 'face_recognized' | 'unknown_face' | 'object_detected' | 'motion'
    label       TEXT,               -- person name / object class
    confidence  REAL,
    source      TEXT,               -- camera id / file name
    image_path  TEXT,
    created_at  TEXT NOT NULL
);
"""


class DBManager:
    """Encapsulates all persistence operations for VisionGuard."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # A plain ":memory:" database is destroyed the instant its single
        # connection closes, which breaks our "open/close per operation"
        # pattern (and unit tests). Using a named, shared-cache in-memory
        # URI keeps the data alive for the lifetime of this DBManager while
        # still requiring no file on disk -- useful for tests/demos.
        self._is_memory = db_path == ":memory:"
        if self._is_memory:
            # A uuid (rather than id(self)) guarantees a unique shared-cache
            # name even if a prior instance was garbage-collected and its
            # id() reused -- avoids cross-test/cross-instance data leaks.
            self._uri = f"file:visionguard_{uuid.uuid4().hex}?mode=memory&cache=shared"
            # Hold one open connection for the whole object's life so the
            # shared in-memory database isn't garbage-collected between calls.
            self._keepalive_conn = sqlite3.connect(self._uri, uri=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        if self._is_memory:
            conn = sqlite3.connect(self._uri, uri=True)
        else:
            conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        logger.info("Database schema ready at %s", self.db_path)

    # ------------------------------------------------------------------
    # Person / face-encoding operations
    # ------------------------------------------------------------------
    def add_person(self, name: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO persons (name, enrolled_on) VALUES (?, ?)",
                (name, datetime.now(timezone.utc).isoformat()),
            )
            if cur.lastrowid:
                return cur.lastrowid
            row = conn.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()
            return row["id"]

    def add_face_encoding(self, person_id: int, encoding_vector) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO face_encodings (person_id, encoding) VALUES (?, ?)",
                (person_id, json.dumps(list(map(float, encoding_vector)))),
            )

    def get_all_encodings(self):
        """Returns list of (person_name, encoding_list) for recognition."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT p.name AS name, f.encoding AS encoding
                   FROM face_encodings f JOIN persons p ON p.id = f.person_id"""
            ).fetchall()
        return [(row["name"], json.loads(row["encoding"])) for row in rows]

    # ------------------------------------------------------------------
    # Event logging (used by analytics_module.logger)
    # ------------------------------------------------------------------
    def log_event(self, event_type: str, label: str = None, confidence: float = None,
                   source: str = None, image_path: str = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO events (event_type, label, confidence, source, image_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event_type, label, confidence, source, image_path, datetime.now(timezone.utc).isoformat()),
            )

    def fetch_events(self, event_type: str = None, limit: int = 500):
        query = "SELECT * FROM events"
        params = ()
        if event_type:
            query += " WHERE event_type = ?"
            params = (event_type,)
        query += " ORDER BY created_at DESC LIMIT ?"
        params += (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
