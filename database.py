"""SQLite database layer for the Campus Vision project."""

from __future__ import annotations

import pickle
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "campus.db"


def ensure_project_directories() -> None:
    """Create runtime folders used by the application."""
    for folder in (
        DATABASE_DIR,
        BASE_DIR / "student_images",
        BASE_DIR / "attendance_reports",
        BASE_DIR / "unknown_captures",
    ):
        folder.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Iterable[sqlite3.Connection]:
    ensure_project_directories()
    connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    """Create all required tables and indexes."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                image_path TEXT NOT NULL,
                encoding BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                camera_source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
                    ON UPDATE CASCADE
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS detection_history (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                name TEXT NOT NULL,
                department TEXT,
                confidence REAL,
                status TEXT NOT NULL,
                camera_source TEXT NOT NULL,
                screenshot_path TEXT,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_attendance_student_date
                ON attendance(student_id, date);
            CREATE INDEX IF NOT EXISTS idx_attendance_date
                ON attendance(date);
            CREATE INDEX IF NOT EXISTS idx_detection_history_date
                ON detection_history(date);
            """
        )


def serialize_encoding(encoding: Any) -> bytes:
    return pickle.dumps(encoding)


def deserialize_encoding(blob: bytes) -> Any:
    return pickle.loads(blob)


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(query, params).fetchall()


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(query, params).fetchone()


def execute(query: str, params: tuple[Any, ...] = ()) -> int:
    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return cursor.lastrowid

