"""Attendance and detection-history logging."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from database import BASE_DIR, execute, fetch_all, fetch_one, initialize_database
from face_recognition_engine import RecognitionResult

try:
    import cv2
except ImportError:
    cv2 = None


REPORTS_DIR = BASE_DIR / "attendance_reports"
UNKNOWN_CAPTURE_DIR = BASE_DIR / "unknown_captures"


class AttendanceLogger:
    def __init__(self, duplicate_minutes: int = 10) -> None:
        initialize_database()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        UNKNOWN_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        self.duplicate_window = timedelta(minutes=duplicate_minutes)

    def log_recognition(
        self,
        result: RecognitionResult,
        camera_source: str,
        frame=None,
    ) -> bool:
        now = datetime.now()
        screenshot_path = None

        if result.status == "unknown" and frame is not None:
            screenshot_path = self.capture_unknown(frame, now)

        self.log_detection_history(result, camera_source, now, screenshot_path)

        if result.status != "recognized" or not result.student_id:
            return False

        if self._is_duplicate(result.student_id, now):
            return False

        execute(
            """
            INSERT INTO attendance (student_id, name, department, date, time, camera_source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.student_id,
                result.name,
                result.department or "",
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                camera_source,
            ),
        )
        return True

    def log_detection_history(
        self,
        result: RecognitionResult,
        camera_source: str,
        detected_at: datetime,
        screenshot_path: str | None = None,
    ) -> None:
        execute(
            """
            INSERT INTO detection_history
                (student_id, name, department, confidence, status, camera_source,
                 screenshot_path, date, time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.student_id,
                result.name,
                result.department,
                result.confidence,
                result.status,
                camera_source,
                screenshot_path,
                detected_at.strftime("%Y-%m-%d"),
                detected_at.strftime("%H:%M:%S"),
            ),
        )

    def capture_unknown(self, frame, detected_at: datetime) -> str:
        if cv2 is None:
            raise ImportError(
                "OpenCV is required to save unknown-person screenshots. "
                "Install it with `pip install opencv-python`."
            )
        filename = f"unknown_{detected_at.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        destination = UNKNOWN_CAPTURE_DIR / filename
        cv2.imwrite(str(destination), frame)
        return str(destination)

    def get_attendance_records(
        self, start_date: str | None = None, end_date: str | None = None, query: str = ""
    ) -> pd.DataFrame:
        sql = "SELECT * FROM attendance WHERE 1=1"
        params: list[str] = []

        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)
        if query.strip():
            sql += " AND (name LIKE ? OR student_id LIKE ? OR department LIKE ?)"
            like_query = f"%{query.strip()}%"
            params.extend([like_query, like_query, like_query])

        sql += " ORDER BY date DESC, time DESC"
        rows = fetch_all(sql, tuple(params))
        return pd.DataFrame([dict(row) for row in rows])

    def get_detection_history(self, limit: int = 100) -> pd.DataFrame:
        rows = fetch_all(
            "SELECT * FROM detection_history ORDER BY detection_id DESC LIMIT ?",
            (limit,),
        )
        return pd.DataFrame([dict(row) for row in rows])

    def export_attendance_csv(self, dataframe: pd.DataFrame | None = None) -> Path:
        if dataframe is None:
            dataframe = self.get_attendance_records()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = REPORTS_DIR / f"attendance_report_{timestamp}.csv"
        dataframe.to_csv(destination, index=False)
        return destination

    def get_department_summary(self) -> pd.DataFrame:
        rows = fetch_all(
            """
            SELECT department, COUNT(*) AS attendance_count
            FROM attendance
            GROUP BY department
            ORDER BY attendance_count DESC
            """
        )
        return pd.DataFrame([dict(row) for row in rows])

    def get_daily_summary(self) -> pd.DataFrame:
        rows = fetch_all(
            """
            SELECT date, COUNT(*) AS attendance_count
            FROM attendance
            GROUP BY date
            ORDER BY date
            """
        )
        return pd.DataFrame([dict(row) for row in rows])

    def _is_duplicate(self, student_id: str, now: datetime) -> bool:
        row = fetch_one(
            """
            SELECT date, time FROM attendance
            WHERE student_id = ?
            ORDER BY attendance_id DESC
            LIMIT 1
            """,
            (student_id,),
        )
        if not row:
            return False

        last_seen = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
        return now - last_seen < self.duplicate_window
