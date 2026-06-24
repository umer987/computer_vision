"""Student registration and administration helpers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from database import (
    BASE_DIR,
    deserialize_encoding,
    execute,
    fetch_all,
    fetch_one,
    initialize_database,
    serialize_encoding,
)
from face_recognition_engine import FaceRecognitionEngine


STUDENT_IMAGES_DIR = BASE_DIR / "student_images"


@dataclass(frozen=True)
class Student:
    student_id: str
    name: str
    department: str
    image_path: str
    encoding: np.ndarray


class StudentManager:
    def __init__(self) -> None:
        initialize_database()
        STUDENT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        self.recognition_engine = FaceRecognitionEngine()

    def add_student(
        self,
        name: str,
        student_id: str,
        department: str,
        image_file: BinaryIO,
        original_filename: str,
    ) -> None:
        self._validate_student_fields(name, student_id, department)
        if fetch_one("SELECT student_id FROM students WHERE student_id = ?", (student_id,)):
            raise ValueError(f"Student ID already exists: {student_id}")

        saved_image_path = self._save_student_image(student_id, image_file, original_filename)
        encoding = self.recognition_engine.extract_single_face_encoding(saved_image_path)

        execute(
            """
            INSERT INTO students (student_id, name, department, image_path, encoding)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                student_id.strip(),
                name.strip(),
                department.strip(),
                str(saved_image_path),
                serialize_encoding(encoding),
            ),
        )

    def update_student(
        self,
        student_id: str,
        name: str,
        department: str,
        image_file: BinaryIO | None = None,
        original_filename: str | None = None,
    ) -> None:
        self._validate_student_fields(name, student_id, department)
        existing = fetch_one("SELECT * FROM students WHERE student_id = ?", (student_id,))
        if not existing:
            raise ValueError(f"No student found with ID: {student_id}")

        image_path = existing["image_path"]
        encoding_blob = existing["encoding"]
        if image_file and original_filename:
            saved_image_path = self._save_student_image(student_id, image_file, original_filename)
            image_path = str(saved_image_path)
            encoding_blob = serialize_encoding(
                self.recognition_engine.extract_single_face_encoding(saved_image_path)
            )

        execute(
            """
            UPDATE students
            SET name = ?, department = ?, image_path = ?, encoding = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE student_id = ?
            """,
            (name.strip(), department.strip(), image_path, encoding_blob, student_id.strip()),
        )

    def delete_student(self, student_id: str) -> None:
        existing = fetch_one("SELECT image_path FROM students WHERE student_id = ?", (student_id,))
        if not existing:
            raise ValueError(f"No student found with ID: {student_id}")

        execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        image_path = Path(existing["image_path"])
        if image_path.exists() and image_path.is_file():
            image_path.unlink()

    def get_student(self, student_id: str) -> Student | None:
        row = fetch_one("SELECT * FROM students WHERE student_id = ?", (student_id,))
        return self._row_to_student(row) if row else None

    def get_all_students(self) -> list[Student]:
        rows = fetch_all("SELECT * FROM students ORDER BY name")
        return [self._row_to_student(row) for row in rows]

    def search_students(self, search_text: str) -> list[Student]:
        query = f"%{search_text.strip()}%"
        rows = fetch_all(
            """
            SELECT * FROM students
            WHERE name LIKE ? OR student_id LIKE ? OR department LIKE ?
            ORDER BY name
            """,
            (query, query, query),
        )
        return [self._row_to_student(row) for row in rows]

    def get_known_face_data(self) -> tuple[list[np.ndarray], list[dict[str, str]]]:
        students = self.get_all_students()
        encodings = [student.encoding for student in students]
        metadata = [
            {
                "student_id": student.student_id,
                "name": student.name,
                "department": student.department,
                "image_path": student.image_path,
            }
            for student in students
        ]
        return encodings, metadata

    def _save_student_image(
        self, student_id: str, image_file: BinaryIO, original_filename: str
    ) -> Path:
        suffix = Path(original_filename).suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError("Only JPG, JPEG, PNG, and WEBP images are supported.")

        destination = STUDENT_IMAGES_DIR / f"{student_id.strip()}{suffix}"
        image_file.seek(0)
        with destination.open("wb") as output:
            shutil.copyfileobj(image_file, output)
        return destination

    @staticmethod
    def _validate_student_fields(name: str, student_id: str, department: str) -> None:
        if not name.strip() or not student_id.strip() or not department.strip():
            raise ValueError("Name, student ID, and department are required.")

    @staticmethod
    def _row_to_student(row) -> Student:
        return Student(
            student_id=row["student_id"],
            name=row["name"],
            department=row["department"],
            image_path=row["image_path"],
            encoding=deserialize_encoding(row["encoding"]),
        )

